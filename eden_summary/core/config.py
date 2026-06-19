import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class WhisperConfig(BaseSettings):
    model: str = Field(default='large-v3', validation_alias='WHISPER_MODEL')
    lang: str | None = Field(default=None, validation_alias='WHISPER_LANGUAGE')
    api_key: str = Field(validation_alias='WHISPER_API_KEY')
    api_base: str = Field(validation_alias='WHISPER_API_BASE')
    chunk_max_chars: int = Field(default=8000, validation_alias='MAX_CHARS')
    chunk_overlap_chars: int = Field(default=1200, validation_alias='CHUNK_OVERLAP_CHARS')
    # Audio longer than this is split into <=N-second pieces before transcription so
    # each request stays well under the ASR API's per-file cap. The pipeline feeds
    # 16 kHz mono WAV (~32 KB/s), so 300 s ≈ 9.6 MB. Kept comfortably below Groq's
    # 25 MB free-tier limit on purpose: uploads near the cap (~19 MB observed) get
    # connection-reset in practice. Self-hosted whisper-server has no cap → raise it.
    asr_chunk_seconds: int = Field(default=300, validation_alias='WHISPER_ASR_CHUNK_SECONDS')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


class LLMConfig(BaseSettings):
    model: str = Field(validation_alias='LLM_MODEL')
    api_key: str = Field(validation_alias='LLM_API_KEY')
    api_base: str | None = Field(default=None, validation_alias='LLM_API_BASE')
    max_retries: int = Field(validation_alias='LLM_MAX_RETRIES')
    temperature: float = Field(validation_alias='LLM_TEMPERATURE')
    timeout: int = Field(validation_alias='LLM_TIMEOUT')
    max_parse_attempts: int = Field(validation_alias='LLM_PARSE_MAX_ATTEMPTS')
    max_workers: int = Field(default=5, validation_alias='LLM_MAX_WORKERS')
    single_pass_token_limit: int = Field(default=32000, validation_alias='LLM_SINGLE_PASS_TOKEN_LIMIT')
    json_mode: bool = Field(default=True, validation_alias='LLM_JSON_MODE')
    # Tier-2 faithfulness judge. Separate model on purpose (summarizer ≠ judge):
    # a non-summarizer model grades the summary against the transcript. judge_api_*
    # are optional overrides (e.g. an OpenRouter endpoint); they fall back to the
    # main LLM credentials when unset.
    judge_enabled: bool = Field(default=True, validation_alias='LLM_JUDGE_ENABLED')
    judge_model: str | None = Field(default=None, validation_alias='LLM_JUDGE_MODEL')
    judge_api_base: str | None = Field(default=None, validation_alias='LLM_JUDGE_API_BASE')
    judge_api_key: str | None = Field(default=None, validation_alias='LLM_JUDGE_API_KEY')
    judge_token_limit: int = Field(default=32000, validation_alias='LLM_JUDGE_TOKEN_LIMIT')
    # Q4 SummQ: QA-based consistency check, an independent signal to the Tier-2
    # claim judge (questions from the summary, answered blind from the transcript,
    # compared deterministically). Reuses the judge model/credentials/token-limit.
    # v1 is compute-and-log: below_threshold is recorded but no regeneration is
    # applied yet.
    summq_enabled: bool = Field(default=True, validation_alias='LLM_SUMMQ_ENABLED')
    summq_consistency_threshold: float = Field(default=0.7, validation_alias='LLM_SUMMQ_THRESHOLD')
    summq_max_questions: int = Field(default=8, validation_alias='LLM_SUMMQ_MAX_QUESTIONS')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model

    @property
    def effective_judge_api_base(self) -> str | None:
        return self.judge_api_base or self.api_base

    @property
    def effective_judge_api_key(self) -> str:
        return self.judge_api_key or self.api_key

class SMTPConfig(BaseSettings):
    host: str = Field(validation_alias='SMTP_HOST')
    port: int = Field(default=587, validation_alias='SMTP_PORT')
    username: str = Field(validation_alias='SMTP_USERNAME')
    password: str = Field(validation_alias='SMTP_PASSWORD')
    sender: str | None = Field(default=None, validation_alias='SMTP_SENDER')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @model_validator(mode='after')
    def set_sender_default(self):
        if not self.sender:
            self.sender = self.username
        return self


class CeleryConfig(BaseSettings):
    redis_password: str = Field(validation_alias='REDIS_PASSWORD')
    timeout_grace: int = Field(default=300, validation_alias='JOB_TIMEOUT_GRACE')
    soft_timeout: int = Field(default=21600, validation_alias='JOB_SOFT_TIMEOUT')
    reaper_stale_seconds: int = Field(default=1800, validation_alias='REAPER_STALE_SECONDS')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @property
    def hard_timeout(self):
        return self.soft_timeout + self.timeout_grace

class DBConfig(BaseSettings):
    host: str = Field(default='localhost', validation_alias='DB_HOST')
    port: int = Field(default=5432, validation_alias='DB_PORT')
    username: str = Field(validation_alias='DB_USERNAME')
    password: str = Field(validation_alias='DB_PASSWORD')
    db:str = Field(validation_alias='DB_NAME')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    def __repr__(self) -> str:
        return f'DBConfig(host={self.host}, port={self.port}, db={self.db})'

    @property
    def db_url(self) -> str:
        return f'postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.db}'

class StorageConfig(BaseSettings):
    endpoint: str = Field(validation_alias='S3_ENDPOINT')
    access_key: str = Field(validation_alias='S3_ACCESS_KEY')
    secret_key: str = Field(validation_alias='S3_SECRET_KEY')
    bucket_name: str = Field(validation_alias='S3_BUCKET')
    region: str = Field(validation_alias='S3_REGION')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

@lru_cache
def get_whisper_cfg():
    return WhisperConfig()

@lru_cache()
def get_llm_cfg():
    return LLMConfig()

@lru_cache()
def get_smtp_cfg():
    return SMTPConfig()

@lru_cache()
def get_celery_cfg():
    return CeleryConfig()

@lru_cache()
def get_db_cfg():
    return DBConfig()

@lru_cache()
def get_storage_cfg():
    return StorageConfig()

@lru_cache()
def get_x_api_key():
    X_API_KEY = os.getenv('X_API_KEY')
    if not X_API_KEY:
        raise ValueError("X_API_KEY is not set")
    return X_API_KEY
