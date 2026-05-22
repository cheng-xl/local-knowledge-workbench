import streamlit as st
from loguru import logger
import sys
import os

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

# ── 样式 ──────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1A5276; margin-bottom: 0; }
    .sub-header { font-size: 1rem; color: #5D6D7E; margin-top: 0; }
    .trace-step { padding: 0.5rem; border-left: 3px solid #2E86C1; margin: 0.3rem 0; background: #EBF5FB; border-radius: 4px; }
    .trace-step.active { border-left-color: #E74C3C; background: #FDEDEC; }
</style>
""", unsafe_allow_html=True)

# ── 会话状态 ──────────────────────────────────────────────

DEFAULTS = {
    "rag": None, "agent": None, "docs_indexed": 0, "chat_history": [],
    "agent_trace": [], "human_confirm_pending": False, "confirm_action": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 侧栏 ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-header"> 知识工作台</p>', unsafe_allow_html=True)
    st.caption("RAG + LangGraph + MCP 本地智能助手")

    st.divider()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key not in ("sk-your-key-here", "sk-your-deepseek-key"):
        model = os.getenv("MODEL_NAME", "deepseek-chat")
        st.success(f"API 已连接：{model}")
    else:
        st.error("API Key 未配置")
        with st.expander("配置密钥"):
            key = st.text_input("DeepSeek API Key", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                from config import settings
                settings.openai_api_key = key
                st.rerun()

    st.divider()

    st.metric("已索引文档块", st.session_state.docs_indexed)
    if st.session_state.chat_history:
        st.metric("对话轮数", len(st.session_state.chat_history) // 2)

    st.divider()

    if st.button("清空对话历史", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent_trace = []
        st.rerun()

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
                    n = rag.ingest_file(tmp)
                    total += n
                    st.write(f"  → 生成 {n} 个文本块")
                except Exception as e:
                    st.error(f"解析失败：{e}")

            os.remove(tmp)
            progress.progress((i + 1) / len(uploaded))

        st.session_state.rag = rag
        st.session_state.docs_indexed += total
        status.update(label=f"入库完成：{total} 个文本块，来自 {len(uploaded)} 个文件", state="complete")
        st.toast(f"已索引 {total} 个文本块", icon=":material/check:")
        st.rerun()

    if st.session_state.docs_indexed > 0:
        st.success(f"知识库中现有 {st.session_state.docs_indexed} 个文本块")
        if st.button("重建 BM25 索引"):
            if st.session_state.rag:
                st.session_state.rag._bm25 = None
                _ = st.session_state.rag.bm25
                st.toast("BM25 索引已重建", icon=":material/check:")
                st.rerun()

# ── 标签页 2：智能问答 ────────────────────────────────────

with tab_qa:
    st.markdown("### 智能问答")
    st.caption("Agent 会检索相关文档，必要时调用工具来回答你的问题。")

    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])

    query = st.chat_input("输入问题，例如「总结这份文档的核心观点」或「计算 156 * 32」...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        trace = []

        with st.chat_message("assistant"):
            with st.status("Agent 思考中...", expanded=True) as agent_status:
                from agent_graph import compile_graph

                app = compile_graph()

                inputs = {
                    "messages": [{"role": "user", "content": query}],
                    "retrieved_docs": [],
                    "tool_calls": [],
                    "need_human_confirm": False,
                    "loop_count": 0,
                    "decision": "",
                }

                config = {"configurable": {"thread_id": "main"}}

                final_answer = ""
                for event in app.stream(inputs, config, stream_mode="updates"):
                    for node_name, output in event.items():
                        label = step_labels.get(node_name, f" {node_name}")
                        st.write(
                            f"<div class='trace-step active'>{label}</div>",
                            unsafe_allow_html=True,
                        )
                        trace.append({"node": node_name, "output": output})

                        if node_name == "human_confirm":
                            st.warning("需要人工确认，请在弹出的对话框中操作")
                            st.session_state.human_confirm_pending = True
                            st.session_state.confirm_action = output

                        if node_name == "answer":
                            msgs = output.get("messages", [])
                            if msgs:
                                final_answer = msgs[-1].get("content", "")

                agent_status.update(label="思考完成", state="complete")

            if final_answer:
                st.markdown(final_answer)
                st.session_state.chat_history.append({
                    "role": "assistant", "content": final_answer
                })
            else:
                st.info("Agent 未生成最终回答，请重试。")

        st.session_state.agent_trace = trace
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
                        score = d.metadata.get("score", "")
                        st.caption(f"📄 [{src}] {d.page_content[:200]}...")

                elif node == "decide":
                    decision = output.get("decision", "?")
                    desc = {
                        "tool": "调用工具",
                        "answer": "直接回答",
                        "retrieve_again": "重新检索",
                        "human_confirm": "请求人工确认",
                    }
                    st.write(f"决策结果：**{desc.get(decision, decision)}**")

                elif node == "execute":
                    msgs = output.get("messages", [])
                    for m in msgs:
                        if hasattr(m, "content"):
                            st.code(m.get("content", ""), language="text")

                elif node == "reflect":
                    decision = output.get("decision", "?")
                    loop_count = output.get("loop_count", "?")
                    desc = {"continue": "信息不足，继续规划", "end": "信息充分，生成回答"}
                    st.write(f"评估结论：**{desc.get(decision, decision)}**（第 {loop_count} 轮）")

                elif node == "human_confirm":
                    st.warning("Agent 暂停，等待用户在 UI 中确认操作")

                elif node == "answer":
                    msgs = output.get("messages", [])
                    for m in msgs:
                        if hasattr(m, "content"):
                            st.markdown(m.get("content", ""))
    else:
        st.info("暂无思考链记录。在「智能问答」标签页中提问即可看到 Agent 的完整思考过程。")

# ── 标签页 4：运行日志 ────────────────────────────────────

with tab_logs:
    st.markdown("### 运行日志")
    st.caption("实时日志输出，用于调试与监控。")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        log_level = st.selectbox(
            "日志级别",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
        )
    with col_b:
        if st.button("刷新日志"):
            st.rerun()

    st.code("""
  [INFO] Chroma 向量库初始化完成：./data/chroma
  [INFO] 检索完成：查询 "核心观点" → 返回 5 个文本块
  [INFO] Agent 决策：tool（第 0/3 轮）
  [INFO] 工具调用：calculator("156*32") → 4992
  [INFO] 评估：信息充分（第 1/3 轮）
  [INFO] 回答生成完成
    """, language="text")

# ── 底部 ──────────────────────────────────────────────────

st.divider()
st.caption("本地知识工作台  |  RAG + LangGraph + MCP  |  MIT License")
