"""Configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_url: str = "http://46.224.162.211:8080"
    llm_model: str = "ll-custom-slerp"

    # Database
    database_url: str = "postgresql+asyncpg://kintsugi:changeme@postgres:5432/kintsugi"

    # Agent
    agent_name: str = "Kintsugi"
    persona_file: str = "/app/persona/megs_persona.md"

    # OGPSA
    ogpsa_enabled: bool = True

    # Ornith
    ornith_scaffold: bool = True

    # Mnemosyne
    mnemosyne_sira_enabled: bool = True
    mnemosyne_enrichment_enabled: bool = True
    consolidation_interval_hours: int = 6

    # HippoRAG
    hipporag_url: str = "http://hipporag:11235"

    # Pharos
    pharos_packs_dir: str = "/app/packs"

    # Garuda
    garuda_enabled: bool = True

    # Network
    cors_origins: str = "*"
    log_level: str = "info"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
