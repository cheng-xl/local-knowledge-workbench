from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM — DeepSeek (OpenAI-compatible)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"

    # SiliconFlow API (Embedding + Reranker)
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    reranker_model: str = "Qwen/Qwen3-VL-Reranker-8B"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma"

    # MCP
    mcp_allowed_path: str = "."

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 128
    retrieval_top_k: int = 5

    # Agent
    max_loops: int = 3

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
