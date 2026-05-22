from typing import TypedDict, List, Annotated, Literal, Optional
import operator
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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
        _llm = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
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


# ── Tool registry ──────────────────────────────────────────

from tools.calculator import calculator
from tools.datetime_tool import datetime_tool

LOCAL_TOOLS = {"calculator": calculator, "datetime_tool": datetime_tool}


# ── Nodes ──────────────────────────────────────────────────


def retrieve_node(state: AgentState) -> dict:
    from rag_pipeline import RAGPipeline

    last_msg = state["messages"][-1]
    query = last_msg.get("content", "") if isinstance(last_msg, dict) else ""

    rag = RAGPipeline()
    docs = rag.hybrid_search(query, use_rerank=True)
    logger.info(f"Retrieved {len(docs)} chunks for query: {query[:80]}...")

    return {"retrieved_docs": docs, "loop_count": 0}


def decide_node(state: AgentState) -> dict:
    msgs = state["messages"]
    last_user = _last_user_msg(msgs)
    query = last_user.get("content", "") if last_user else ""
    docs = state.get("retrieved_docs", [])
    loop_count = state.get("loop_count", 0)
    tool_msgs = _tool_results(msgs)

    doc_summary = "\n".join(
        f"[{i+1}] {d.metadata.get('source_file', '?')}: {d.page_content[:200]}..."
        for i, d in enumerate(docs[:5])
    )

    prompt = f"""You are an AI assistant deciding the next action.

User query: {query}

Retrieved documents ({len(docs)} total):
{doc_summary or '(none)'}

Tool results so far ({len(tool_msgs)}):
{[m.get('content', '')[:300] for m in tool_msgs[-3:]] or '(none)'}

Loop count: {loop_count}/{MAX_LOOPS}

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

    logger.info(f"Decide: {decision} (loop {loop_count}/{MAX_LOOPS})")
    return {"decision": decision}


def execute_node(state: AgentState) -> dict:
    if state.get("decision") != "tool":
        return {}

    msgs = state["messages"]
    last_user = _last_user_msg(msgs)

    prompt = f"""You have access to these tools: {', '.join(LOCAL_TOOLS.keys())}
User request: {last_user.get('content', '') if last_user else 'No query'}

If the user wants a calculation, respond with: calculator("expression")
If the user wants date/time, respond with: datetime_tool("action")
Otherwise respond: none

Respond with ONLY the tool call in the exact format above, nothing else."""

    response = _call_llm(prompt)
    text = response.strip()
    logger.info(f"Execute LLM response: {text[:100]}")

    result_msg = None
    for name in LOCAL_TOOLS:
        if name in text:
            try:
                import re
                match = re.search(rf'{name}\(\s*"([^"]*)"\s*\)', text)
                arg = match.group(1) if match else ""
                tool_fn = LOCAL_TOOLS[name]
                result = tool_fn(arg)
                result_msg = {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": f"call_{name}",
                    "name": name,
                }
                logger.info(f"Tool {name}({arg}) -> {str(result)[:80]}")
            except Exception as e:
                result_msg = {
                    "role": "tool",
                    "content": f"Error calling {name}: {e}",
                    "tool_call_id": f"call_{name}",
                    "name": name,
                }
            break

    if result_msg is None:
        result_msg = {
            "role": "tool",
            "content": text,
            "tool_call_id": "call_unknown",
            "name": "unknown",
        }

    return {"messages": [result_msg], "tool_calls": []}


def reflect_node(state: AgentState) -> dict:
    loop_count = state.get("loop_count", 0) + 1

    if loop_count >= MAX_LOOPS:
        logger.warning(f"Max loops ({MAX_LOOPS}) reached, forcing end")
        return {"loop_count": loop_count, "decision": "end"}

    msgs = state["messages"]
    docs = state.get("retrieved_docs", [])
    last_user = _last_user_msg(msgs)
    tool_msgs = _tool_results(msgs)

    prompt = f"""Evaluate whether we have enough information to answer the user's question.

User query: {last_user.get('content', '') if last_user else 'N/A'}
Documents retrieved: {len(docs)}
Tool results: {len(tool_msgs)}
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
    return {"need_human_confirm": True}


def answer_node(state: AgentState) -> dict:
    msgs = state["messages"]
    docs = state.get("retrieved_docs", [])
    last_user = _last_user_msg(msgs)
    tool_msgs = _tool_results(msgs)

    context = "\n\n".join(
        f"[Doc {i+1} from {d.metadata.get('source_file', '?')}]\n{d.page_content}"
        for i, d in enumerate(docs[:5])
    )

    tools_text = "\n".join(
        f"[{m.get('name', '?')}] {m.get('content', '')}"
        for m in tool_msgs
    )

    user_query = last_user.get("content", "") if last_user else "(no query)"

    answer_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

USER QUESTION:
{user_query}

CONTEXT DOCUMENTS:
{context or '(no documents retrieved)'}

TOOL RESULTS:
{tools_text or '(no tools called)'}

Generate a comprehensive answer in Chinese. If the context doesn't contain relevant information, answer directly based on your knowledge. If it's a simple calculation or factual question, just answer it."""

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
        "tool": "execute",
        "answer": "answer",
        "human_confirm": "human_confirm",
        "retrieve_again": "retrieve",
    })

    graph.add_edge("execute", "reflect")

    graph.add_conditional_edges("reflect", should_continue, {
        "continue": "decide",
        "end": "answer",
    })

    graph.add_edge("human_confirm", "execute")
    graph.add_edge("answer", END)

    return graph


def compile_graph(checkpointer=None):
    graph = build_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
