import os

# 必须在所有第三方导入之前设置，确保 huggingface_hub/transformers 使用镜像
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM — DeepSeek (OpenAI-compatible)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    hf_endpoint: str = "https://hf-mirror.com"

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
