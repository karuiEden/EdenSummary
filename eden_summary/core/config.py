import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class WhisperConfig(BaseSettings):
    model: str = Field(default='large-v3', validation_alias='WHISPER_MODEL')
    lang: str | None = Field(default=None, validation_alias='WHISPER_LANGUAGE')
    device: str = Field(default='auto', validation_alias='WHISPER_DEVICE')
    compute_type: str = Field(default='auto' ,validation_alias='WHISPER_COMPUTE_TYPE')
    chunk_max_chars: int = Field(default=4000, validation_alias='MAX_CHARS')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


class LLMConfig(BaseSettings):
    model: str = Field(validation_alias='LLM_MODEL')
    api_key: str = Field(validation_alias='LLM_API_KEY')
    api_base: str | None = Field(default=None, validation_alias='LLM_API_BASE')
    max_retries: int = Field(validation_alias='LLM_MAX_RETRIES')
    temperature: float = Field(validation_alias='LLM_TEMPERATURE')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

class SMTPConfig(BaseSettings):
    host: str = Field(validation_alias='SMTP_HOST')
    port: int = Field(default=587, validation_alias='SMTP_PORT')
    username: str = Field(validation_alias='SMTP_USERNAME')
    password: str = Field(validation_alias='SMTP_PASSWORD')
    sender: str = Field(default=None, validation_alias='SMTP_SENDER')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    @model_validator(mode='after')
    def set_sender_default(self):
        if not self.sender:
            self.sender = self.username
        return self


class CeleryConfig(BaseSettings):
    redis_password: str = Field(validation_alias='REDIS_PASSWORD')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

class DBConfig(BaseSettings):
    host: str = Field(default='localhost', validation_alias='DB_HOST')
    port: int = Field(default=5432, validation_alias='DB_PORT')
    username: str = Field(validation_alias='DB_USERNAME')
    password: str = Field(validation_alias='DB_PASSWORD')
    db:str = Field(validation_alias='DB_NAME')
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

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
