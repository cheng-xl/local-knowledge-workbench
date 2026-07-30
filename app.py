import streamlit as st
from loguru import logger
import sys
import os
import io

# Load .env into os.environ
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="本地知识工作台",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 日志 ──────────────────────────────────────────────────

logger.remove()

logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan> | {message}",
)

# StringIO sink — re-add on every rerun (logger.remove() wipes all sinks)
if "log_stream" not in st.session_state:
    st.session_state.log_stream = io.StringIO()
logger.add(
    st.session_state.log_stream,
    level="DEBUG",
    format="{time:HH:mm:ss} | {level:7} | {name}:{line} | {message}",
)

# ── 样式 ──────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── 锁定整体页面，禁止外层滚动 ── */
    html, body, #root {
        margin: 0;
        padding: 0;
        height: 100%;
        overflow: hidden;
    }
    [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh;
        max-height: 100vh;
    }
    [data-testid="stAppViewContainer"] > section {
        overflow: hidden !important;
    }
    .main .block-container {
        max-height: 100vh;
        height: 100vh;
        overflow: hidden !important;
        padding: 0.4rem 1.5rem 0 1.5rem;
        display: flex;
        flex-direction: column;
    }
    .main .block-container > div:first-child {
        flex: 1;
        min-height: 0;
    }
    /* ── 隐藏 Streamlit 表单默认提示 ── */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    /* ── 标签页填满可用高度 ── */
    .stTabs {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }
    .stTabs > div:first-child {
        flex-shrink: 0;
    }
    /* ── 各标签面板独立滚动 ── */
    .stTabs [role="tabpanel"] {
        flex: 1;
        min-height: 0;
        overflow-y: auto !important;
        padding-right: 6px;
    }
    /* ── 侧栏紧凑 ── */
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
        overflow-y: auto;
    }
    /* ── 思考链样式 ── */
    .main-header { font-size: 2rem; font-weight: 700; color: #1A5276; margin-bottom: 0; }
    .trace-step { padding: 0.5rem; border-left: 3px solid #2E86C1; margin: 0.3rem 0; background: #EBF5FB; border-radius: 4px; }
    .trace-step.active { border-left-color: #E74C3C; background: #FDEDEC; }
</style>
""", unsafe_allow_html=True)

# ── 会话状态 ──────────────────────────────────────────────

DEFAULTS = {
    "rag": None, "docs_indexed": 0, "chat_history": [],
    "agent_trace": [], "thread_id": None,
    "pending_interrupt": False, "pending_config": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Lazy-load heavy modules
from config import settings


@st.cache_resource
def get_compiled_graph():
    from agent_graph import compile_graph
    return compile_graph()


@st.cache_resource
def get_vector_store():
    from vector_store import _get_global_vs
    return _get_global_vs()


# ── 侧栏（版本信息移至此） ──

with st.sidebar:
    st.markdown('<p class="main-header"> 知识工作台</p>', unsafe_allow_html=True)
    st.caption("RAG + LangGraph + MCP 本地智能助手")
    st.divider()

    api_key = settings.openai_api_key
    if api_key and api_key not in ("sk-your-key-here", "sk-your-deepseek-key"):
        st.success(f"API 已连接：{settings.model_name}")
    else:
        st.error("API Key 未配置")
        with st.expander("配置密钥"):
            key = st.text_input("DeepSeek API Key", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                settings.openai_api_key = key
                st.rerun()

    st.divider()

    vs = get_vector_store()
    total_docs = vs._collection.count()
    st.metric("已索引文档块", total_docs)
    if st.session_state.chat_history:
        st.metric("对话轮数", len(st.session_state.chat_history) // 2)

    st.divider()
    if st.button("清空对话历史", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_trace = []
        st.session_state.thread_id = None
        st.rerun()
    st.divider()
    st.caption("RAG + LangGraph + MCP | MIT")

# ── 标签页 ────────────────────────────────────────────────

step_labels = {
    "retrieve": "检索相关文档",
    "decide": "决策：判断下一步动作",
    "execute": "执行工具调用",
    "reflect": "评估执行结果",
    "human_confirm": "等待人工确认",
    "answer": "生成最终回答",
}

tab_docs, tab_qa, tab_trace, tab_logs = st.tabs([
    " 文档上传", " 智能问答", " 思考链追踪", " 运行日志"
])

# ── 标签页 1：文档上传 ────────────────────────────────────

with tab_docs:
    st.markdown("### 文档入库")
    st.caption("上传文档构建知识库，支持 PDF、Word、Markdown、TXT 格式。")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "拖拽或选择文件",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    with col2:
        chunk_size = st.slider("分块大小", 256, 1024, 512, step=64)
        overlap = st.slider("重叠字数", 32, 256, 128, step=32)

    if uploaded and st.button("开始入库", type="primary", use_container_width=True):
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        progress = st.progress(0)
        status = st.status("正在处理文档...")
        total = 0
        for i, f in enumerate(uploaded):
            tmp = f"/tmp/{f.name}"
            os.makedirs("/tmp", exist_ok=True)
            with open(tmp, "wb") as fp:
                fp.write(f.getbuffer())
            with status:
                st.write(f"解析中：{f.name}")
                try:
                    n = rag.ingest_file(tmp, chunk_size=chunk_size, overlap=overlap)
                    total += n
                    st.write(f"  → 生成 {n} 个文本块")
                except Exception as e:
                    st.error(f"解析失败：{e}")
            os.remove(tmp)
            progress.progress((i + 1) / len(uploaded))
        st.session_state.rag = rag
        st.session_state.docs_indexed += total
        get_vector_store.clear()
        status.update(label=f"入库完成：{total} 个文本块，来自 {len(uploaded)} 个文件", state="complete")
        st.toast(f"已索引 {total} 个文本块", icon=":material/check:")
        st.rerun()

    # ── 知识库管理 ──────────────────────────────────────
    st.divider()
    st.markdown("### 知识库管理")
    vs = get_vector_store()
    stats = vs.get_stats()

    if stats["total"] == 0:
        st.info("知识库为空，请上传文档。")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("文档块总数", stats["total"])
        with col_b:
            st.metric("来源文件数", len(stats["sources"]))
        with col_c:
            if st.button("重建 BM25 索引", use_container_width=True):
                if st.session_state.rag:
                    st.session_state.rag._bm25 = None
                    _ = st.session_state.rag.bm25
                    st.toast("BM25 索引已重建", icon=":material/check:")
                    st.rerun()

        st.markdown("**来源文件列表：**")
        for fname, count in stats["sources"]:
            c1, c2, c3 = st.columns([5, 2, 2])
            with c1:
                st.text(f"{fname}")
            with c2:
                st.caption(f"{count} 块")
            with c3:
                if st.button("删除", key=f"del_{fname}_{count}",
                             use_container_width=True):
                    n = vs.delete_by_source(fname)
                    st.session_state.docs_indexed = max(0, st.session_state.docs_indexed - n)
                    if st.session_state.rag:
                        st.session_state.rag._bm25 = None
                        st.session_state.rag._doc_texts = []
                    get_vector_store.clear()
                    st.toast(f"已删除「{fname}」({n} 块)", icon=":material/delete:")
                    st.rerun()

# ── 标签页 2：智能问答 ────────────────────────────────────

with tab_qa:
    st.markdown("### 智能问答")
    st.caption("Agent 会检索相关文档，必要时调用工具来回答你的问题。对话历史自动保留。")

    col_chat, col_input = st.columns([3, 1])

    # ── 左侧：对话区 ──
    with col_chat:
        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg["content"])

    # ── 右侧：输入区 + 快捷提示 + 人工确认 ──
    with col_input:
        with st.form("qa_form", clear_on_submit=True):
            query = st.text_area(
                "输入问题",
                placeholder="例如「总结这份文档」或「计算 156 * 32」",
                height=150,
                label_visibility="collapsed",
            )
            st.caption("Ctrl+Enter 提交")
            submitted = st.form_submit_button("🚀 发送", type="primary", use_container_width=True)
        query = query if submitted else None
        # 快捷提问
        with st.expander("快捷提问"):
            for shortcut in ["总结这份文档的核心观点", "这份文档讲了什么？",
                             "计算 156 * 32", "今天是几号？星期几？"]:
                if st.button(shortcut, use_container_width=True, key=f"shortcut_{shortcut}"):
                    query = shortcut

        # Human-in-the-Loop
        human_confirm_action = None
        if st.session_state.get("pending_interrupt"):
            st.divider()
            st.warning("Agent 请求执行敏感操作")
            if st.button("✅ 批准操作", use_container_width=True, type="primary"):
                human_confirm_action = True
            if st.button("❌ 拒绝操作", use_container_width=True):
                human_confirm_action = False

    # ── 处理逻辑（spinner 在右侧，不向左侧写任何内容）──
    if query or human_confirm_action is not None:
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
        trace = []
        final_answer = ""

        with col_input:
            with st.spinner("思考中..."):
                try:
                    app = get_compiled_graph()

                    if st.session_state.thread_id is None:
                        import uuid
                        st.session_state.thread_id = str(uuid.uuid4())
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}

                    # Resume from pending interrupt
                    if human_confirm_action is not None and st.session_state.get("pending_interrupt"):
                        from langgraph.types import Command
                        st.session_state.pending_interrupt = False
                        st.session_state.pending_config = None
                        for event in app.stream(
                            Command(resume=human_confirm_action), config,
                            stream_mode="updates",
                        ):
                            for node_name, output in event.items():
                                trace.append({"node": node_name, "output": output})
                                if node_name == "answer":
                                    msgs = output.get("messages", [])
                                    if msgs:
                                        final_answer = msgs[-1].get("content", "")
                    elif query:
                        inputs = {
                            "messages": [{"role": "user", "content": query}],
                            "retrieved_docs": [],
                            "tool_calls": [],
                            "need_human_confirm": False,
                            "loop_count": 0,
                            "decision": "",
                        }
                        for event in app.stream(inputs, config, stream_mode="updates"):
                            for node_name, output in event.items():
                                trace.append({"node": node_name, "output": output})
                                if node_name == "answer":
                                    msgs = output.get("messages", [])
                                    if msgs:
                                        final_answer = msgs[-1].get("content", "")

                except Exception as e:
                    from langgraph.errors import GraphInterrupt
                    if isinstance(e, GraphInterrupt):
                        st.session_state.pending_interrupt = True
                        st.session_state.pending_config = config
                        st.warning("Agent 请求执行敏感操作，请确认")
                        st.rerun()
                    else:
                        logger.exception("Agent execution failed")
                        final_answer = f"抱歉，处理请求时出错：{e}"

        if final_answer:
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

        st.session_state.agent_trace = trace
        if not st.session_state.get("pending_interrupt"):
            st.rerun()

# ── 标签页 3：思考链追踪 ──────────────────────────────────

with tab_trace:
    st.markdown("### Agent 思考链")
    st.caption("展示 LangGraph Agent 每一步的决策与执行过程。")

    if st.session_state.agent_trace:
        for i, step in enumerate(st.session_state.agent_trace):
            with st.expander(
                f"第 {i+1} 步：{step_labels.get(step['node'], step['node'])}",
                expanded=(i == len(st.session_state.agent_trace) - 1),
            ):
                node = step["node"]
                output = step["output"]

                if node == "retrieve":
                    docs = output.get("retrieved_docs", [])
                    st.write(f"检索到 {len(docs)} 篇相关文档")
                    for d in docs[:5]:
                        src = d.metadata.get("source_file", "未知来源")
                        st.caption(f"📄 [{src}] {d.page_content[:200]}...")

                elif node == "decide":
                    decision = output.get("decision", "?")
                    desc = {"tool": "调用工具", "answer": "直接回答",
                            "retrieve_again": "重新检索", "human_confirm": "请求人工确认"}
                    st.write(f"决策结果：**{desc.get(decision, decision)}**")

                elif node == "execute":
                    for m in output.get("messages", []):
                        st.code(m.get("content", ""), language="text")

                elif node == "reflect":
                    decision = output.get("decision", "?")
                    lc = output.get("loop_count", "?")
                    desc = {"continue": "信息不足，继续规划", "end": "信息充分，生成回答"}
                    st.write(f"评估结论：**{desc.get(decision, decision)}**（第 {lc} 轮）")

                elif node == "human_confirm":
                    st.warning("Agent 暂停，等待用户在 UI 中确认操作")

                elif node == "answer":
                    for m in output.get("messages", []):
                        st.markdown(m.get("content", ""))
    else:
        st.info("暂无思考链记录。在「智能问答」标签页中提问即可看到 Agent 的完整思考过程。")

# ── 标签页 4：运行日志 ────────────────────────────────────

with tab_logs:
    st.markdown("### 运行日志")
    st.caption("实时日志输出，用于调试与监控。")
    if st.button("刷新日志", use_container_width=True):
        st.rerun()
    log_text = st.session_state.log_stream.getvalue()
    if log_text:
        st.code(log_text[-5000:], language="log")
    else:
        st.info("暂无日志。等一下提问或上传文档就会有新的日志产生。")

# ── 底部 ──


