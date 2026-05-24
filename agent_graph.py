import re
import operator
from typing import TypedDict, List, Annotated, Optional
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from shared import Document
from config import settings
from loguru import logger


# ── State ──────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    retrieved_docs: List[Document]
    tool_calls: List[dict]
    need_human_confirm: bool
    loop_count: int
    decision: str


MAX_LOOPS = settings.max_loops

# ── LLM ────────────────────────────────────────────────────

_llm: Optional[OpenAI] = None


def get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(api_key=settings.openai_api_key,
                      base_url=settings.openai_base_url)
    return _llm


def _call_llm(prompt: str) -> str:
    resp = get_llm().chat.completions.create(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _last_user_msg(messages: list) -> Optional[dict]:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return m
    return None


def _tool_results(messages: list) -> list:
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def _node_context(state: AgentState) -> dict:
    """Extract commonly used values from state for downstream nodes."""
    msgs = state["messages"]
    # Build conversation history string for prompts
    history_parts = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        if role == "user":
            history_parts.append(f"用户: {content}")
        elif role == "assistant":
            history_parts.append(f"助手: {content[:200]}")
    history_str = "\n".join(history_parts[-10:])  # last 10 turns
    return {
        "msgs": msgs,
        "last_user": _last_user_msg(msgs),
        "docs": state.get("retrieved_docs", []),
        "tool_msgs": _tool_results(msgs),
        "loop_count": state.get("loop_count", 0),
        "history": history_str,
    }


# ── Tool registry ──────────────────────────────────────────

from tools.calculator import calculator
from tools.datetime_tool import datetime_tool

LOCAL_TOOLS = {"calculator": calculator, "datetime_tool": datetime_tool}

# Keyword-based tool routing (avoids extra LLM call)
TOOL_KEYWORDS = {
    "calculator": ["计算", "算", "+", "-", "*", "/", "×", "÷", "平方", "开方", "等于"],
    "datetime_tool": ["时间", "日期", "今天", "现在", "星期", "几点", "几月", "几号"],
}


def _pick_tool(query: str) -> Optional[str]:
    """Pick tool by keyword matching. Returns tool name or None."""
    for name, keywords in TOOL_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return name
    return None


# ── Nodes ──────────────────────────────────────────────────


def retrieve_node(state: AgentState) -> dict:
    from rag_pipeline import RAGPipeline
    from vector_store import _get_global_vs

    last_msg = state["messages"][-1]
    query = last_msg.get("content", "") if isinstance(last_msg, dict) else ""

    rag = RAGPipeline(vector_store=_get_global_vs())
    docs = rag.hybrid_search(query, use_rerank=True)
    logger.info(f"Retrieved {len(docs)} chunks for query: {query[:80]}...")

    return {"retrieved_docs": docs, "loop_count": 0}


def decide_node(state: AgentState) -> dict:
    ctx = _node_context(state)
    query = ctx["last_user"].get("content", "") if ctx["last_user"] else ""

    doc_summary = "\n".join(
        f"[{i+1}] {d.metadata.get('source_file', '?')}: {d.page_content[:200]}..."
        for i, d in enumerate(ctx["docs"][:5])
    )

    prompt = f"""You are an AI assistant deciding the next action.

CONVERSATION HISTORY:
{ctx['history'] or '(new conversation)'}

Current user query: {query}

Retrieved documents ({len(ctx['docs'])} total):
{doc_summary or '(none)'}

Tool results so far ({len(ctx['tool_msgs'])}):
{[m.get('content', '')[:300] for m in ctx['tool_msgs'][-3:]] or '(none)'}

Loop count: {ctx['loop_count']}/{MAX_LOOPS}

Decide the next step. Reply with EXACTLY ONE WORD:
- tool: need to call a tool (calculator, datetime_tool)
- answer: have enough info to answer the user
- retrieve_again: need to search with a different query
- human_confirm: need user permission for a sensitive operation"""

    response = _call_llm(prompt)
    decision = response.strip().lower()

    valid = {"tool", "answer", "retrieve_again", "human_confirm"}
    if decision not in valid:
        logger.warning(f"LLM returned invalid decision '{decision}', defaulting to answer")
        decision = "answer"

    logger.info(f"Decide: {decision} (loop {ctx['loop_count']}/{MAX_LOOPS})")
    return {"decision": decision}


def execute_node(state: AgentState) -> dict:
    if state.get("decision") != "tool":
        return {}

    ctx = _node_context(state)
    query = ctx["last_user"].get("content", "") if ctx["last_user"] else ""

    # Keyword-based tool selection (avoid LLM call for simple cases)
    tool_name = _pick_tool(query)
    if tool_name is None:
        tool_name = "calculator"  # default fallback

    tool_fn = LOCAL_TOOLS.get(tool_name)
    if tool_fn is None:
        return {"messages": [{"role": "tool", "content": f"Unknown tool: {tool_name}",
                              "tool_call_id": "call_unknown", "name": "unknown"}]}

    # Extract argument: for calculator, find math expression; for datetime, find action
    if tool_name == "calculator":
        # Extract whatever comes after "计算" or just take the full query
        arg = query
        for sep in ["计算", "算一下", "算"]:
            if sep in query:
                arg = query.split(sep, 1)[-1].strip()
                break
    else:
        arg = "now"  # datetime_tool default

    try:
        result = tool_fn(arg)
        logger.info(f"Tool {tool_name}({arg}) -> {str(result)[:80]}")
    except Exception as e:
        result = f"Error: {e}"

    result_msg = {"role": "tool", "content": str(result),
                  "tool_call_id": f"call_{tool_name}", "name": tool_name}
    return {"messages": [result_msg], "tool_calls": []}


def reflect_node(state: AgentState) -> dict:
    loop_count = state.get("loop_count", 0) + 1

    if loop_count >= MAX_LOOPS:
        logger.warning(f"Max loops ({MAX_LOOPS}) reached, forcing end")
        return {"loop_count": loop_count, "decision": "end"}

    ctx = _node_context(state)

    prompt = f"""Evaluate whether we have enough information to answer the user's question.

User query: {ctx['last_user'].get('content', '') if ctx['last_user'] else 'N/A'}
Documents retrieved: {len(ctx['docs'])}
Tool results: {len(ctx['tool_msgs'])}
Loop: {loop_count}/{MAX_LOOPS}

Reply with EXACTLY ONE WORD:
- continue: need more information, go back to decide
- end: have enough to answer"""

    response = _call_llm(prompt)
    decision = response.strip().lower()

    if decision not in ("continue", "end"):
        decision = "end"
    logger.info(f"Reflect: {decision} (loop {loop_count}/{MAX_LOOPS})")

    return {"loop_count": loop_count, "decision": decision}


def human_node(state: AgentState) -> dict:
    logger.info("Human-in-the-Loop: awaiting user confirmation")
    # Pause execution, return user's decision via interrupt
    approved = interrupt("需要人工确认此操作，请在 UI 中批准或拒绝。")
    decision = "approved" if approved else "denied"
    logger.info(f"Human-in-the-Loop: user {decision}")
    return {"decision": decision, "need_human_confirm": False}


def answer_node(state: AgentState) -> dict:
    ctx = _node_context(state)

    context = "\n\n".join(
        f"[Doc {i+1} from {d.metadata.get('source_file', '?')}]\n{d.page_content}"
        for i, d in enumerate(ctx["docs"][:5])
    )

    tools_text = "\n".join(
        f"[{m.get('name', '?')}] {m.get('content', '')}"
        for m in ctx["tool_msgs"]
    )

    user_query = ctx["last_user"].get("content", "") if ctx["last_user"] else "(no query)"

    answer_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

CONVERSATION HISTORY:
{ctx['history'] or '(new conversation)'}

USER QUESTION:
{user_query}

CONTEXT DOCUMENTS:
{context or '(no documents retrieved)'}

TOOL RESULTS:
{tools_text or '(no tools called)'}

Generate a comprehensive answer in Chinese. Reference the conversation history when the user asks follow-up questions (like "add 3" should refer to the previous calculation). If the context doesn't contain relevant information, answer directly based on your knowledge."""

    response = _call_llm(answer_prompt)
    logger.info(f"Answer generated: {response[:100]}...")

    return {"messages": [{"role": "assistant", "content": response}],
            "need_human_confirm": False}


# ── Router functions ───────────────────────────────────────

def route_after_decide(state: AgentState) -> str:
    return state.get("decision", "answer")


def should_continue(state: AgentState) -> str:
    return state.get("decision", "end")


# ── Graph construction ─────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_node("execute", execute_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("human_confirm", human_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges("decide", route_after_decide, {
        "tool": "execute", "answer": "answer",
        "human_confirm": "human_confirm", "retrieve_again": "retrieve",
    })
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges("reflect", should_continue, {
        "continue": "decide", "end": "answer",
    })
    graph.add_edge("human_confirm", "execute")
    graph.add_edge("answer", END)

    return graph


def compile_graph(checkpointer=None):
    graph = build_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
