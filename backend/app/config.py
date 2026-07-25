"""Application settings.

Values can be overridden with environment variables (prefix ``REGLENS_``),
e.g. ``REGLENS_CHAT_MODEL_ALIAS=phi-4-mini``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGLENS_")

    # --- Foundry Local models -------------------------------------------------
    # phi-3.5-mini is the default because this project targets CPU-only laptops:
    # it is ~2.5 GB versus ~4.8 GB for phi-4-mini and answers noticeably faster
    # while staying coherent on English regulatory text. Set
    # REGLENS_CHAT_MODEL_ALIAS=phi-4-mini if you have the headroom.
    chat_model_alias: str = "phi-3.5-mini"
    embedding_model_alias: str = "qwen3-embedding-0.6b"
    # Some machines (including the reference dev machine) cannot initialise the
    # OpenVINO GPU execution provider, so the CPU variant is selected by default.
    force_cpu_variant: bool = True

    # Foundry Local defaults its model cache to ~/.{app_name}/cache/models, which
    # means every app that uses the SDK downloads its own multi-gigabyte copy of
    # the same models. Pointing all projects at one shared directory avoids that.
    # Override with REGLENS_MODEL_CACHE_DIR.
    model_cache_dir: Path = Path.home() / ".foundry-shared" / "cache" / "models"

    # --- Storage --------------------------------------------------------------
    db_path: Path = DATA_DIR / "reglens.db"
    upload_dir: Path = DATA_DIR / "documents"

    # --- Upload limits --------------------------------------------------------
    max_file_size_mb: int = 10
    max_document_count: int = 20
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")

    # --- Chunking -------------------------------------------------------------
    max_chunk_chars: int = 1600
    chunk_overlap_chars: int = 200

    # --- Retrieval ------------------------------------------------------------
    top_k: int = 6
    # Weights for the hybrid score. Semantic similarity carries most of the
    # signal; BM25 rescues exact legal phrasing ("post-market surveillance")
    # that embeddings alone can blur together.
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    structure_weight: float = 0.1

    # --- Generation -----------------------------------------------------------
    max_answer_tokens: int = 700


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
