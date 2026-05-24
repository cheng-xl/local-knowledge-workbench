# Local Knowledge Workbench

基于 **RAG + LangGraph + MCP** 的本地智能知识工作台，支持文档问答、工具调用和 Human-in-the-Loop 人工确认。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 功能

- **文档入库与检索**：支持 PDF、Word、Markdown、TXT，混合检索（向量 + BM25 + Cross-encoder 重排序）
- **Agent 智能问答**：LangGraph 6 节点状态图，支持检索→决策→工具执行→评估→回答
- **工具调用**：内置计算器、日期时间工具，可扩展
- **MCP 协议集成**：自建只读 Filesystem MCP Server，展示标准化工具调用
- **Human-in-the-Loop**：敏感操作需人工确认后才能执行
- **对话记忆**：多轮对话上下文自动保留
- **中文界面**：Streamlit 4 标签页 UI（文档管理 / 问答 / 思考链 / 日志）

## 架构

```
Streamlit UI
    │
LangGraph Agent（6 节点状态图）
    │
    ├── RAG Pipeline（向量 + BM25 + RRF + 重排序）
    ├── MCP Server（只读文件系统）
    └── Tools（计算器、日期时间）
    │
Chroma 向量数据库 + BGE Embedding
```

**Agent 状态流转：**
```
retrieve → decide → execute → reflect → answer
             │                     ↑
             ├── human_confirm ────┘（需人工确认时暂停）
             └── retrieve_again ─────→（重新检索）
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 8GB RAM（本地运行 Embedding 模型）
- Windows / macOS / Linux

### 2. 安装

```bash
# 克隆仓库
git clone https://github.com/cheng-xl/local-knowledge-workbench.git
cd local-knowledge-workbench

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 默认使用 DeepSeek API（兼容 OpenAI 协议）
```

`.env` 示例：
```bash
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
HF_ENDPOINT=https://hf-mirror.com   # 国内用户需要 HuggingFace 镜像
```

### 4. 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

首次启动会自动下载 Embedding 模型（约 400MB），请耐心等待。

### 5. 使用

1. **上传文档**：在「文档上传」标签页拖入 PDF/Word/Markdown/TXT 文件
2. **提问**：切换到「智能问答」，输入问题
3. **查看思考链**：在「思考链追踪」查看 Agent 每一步的决策过程
4. **监控日志**：在「运行日志」查看实时系统日志

## 运行测试

```bash
pytest tests/ -v
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph（6 节点状态图、条件路由、checkpoint） |
| LLM | DeepSeek / OpenAI（兼容 OpenAI 协议的 API） |
| Embedding | BAAI/bge-small-zh-v1.5（通过 sentence-transformers 本地运行） |
| 向量数据库 | Chroma（本地持久化） |
| 检索 | BM25（rank-bm25）+ 向量检索 + RRF 融合 + Cross-encoder 重排序 |
| MCP 协议 | 自建 MCP Server（read_file、list_dir），stdio 通信 |
| UI | Streamlit（4 标签页中文界面） |
| 文档处理 | pypdf、python-docx |
| 日志 | loguru（stderr + Streamlit 内嵌日志面板） |
| 配置 | pydantic-settings + python-dotenv |
| 测试 | pytest（33 个测试） |

## 项目结构

```
├── app.py                 # Streamlit 入口（UI + 会话管理）
├── agent_graph.py         # LangGraph Agent（状态、节点、路由）
├── rag_pipeline.py        # RAG 全流程（加载、分块、检索、重排序）
├── vector_store.py        # Chroma 向量库封装
├── mcp_server.py          # MCP Filesystem Server
├── shared.py              # 共享类型（Document）
├── config.py              # 配置中心（pydantic-settings）
├── requirements.txt       # Python 依赖
├── .env.example           # 配置模板
├── tools/
│   ├── calculator.py      # 安全数学表达式求值
│   └── datetime_tool.py   # 日期时间查询
├── tests/
│   ├── test_agent.py      # Agent 图结构 + 路由测试
│   ├── test_rag.py        # RAG 分块 + 融合测试
│   └── test_scenarios.py  # 工具 + MCP 测试
└── data/chroma/           # Chroma 持久化数据（gitignore）
```

## License

MIT
