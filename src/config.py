"""Configuration settings for LLM Multi-Head Attention Lab."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Attention demo."""

    app_name: str = "LLM Multi-Head Self-Attention Lab"
    app_version: str = "0.1.0"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    presets_file: Path = Path(__file__).resolve().parent.parent / "data" / "presets.json"

    default_head_count: int = 4
    default_d_k: int = 8
    default_temperature: float = 1.0
    default_causal: bool = False

    model_config = SettingsConfigDict(
        env_prefix="ATTENTION_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
