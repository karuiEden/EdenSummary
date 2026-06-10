from functools import lru_cache
from pathlib import Path

from boto3 import client
from botocore.config import Config

from eden_summary.core import get_storage_cfg


@lru_cache(maxsize=None)
def _get_s3() -> tuple:
    cfg = get_storage_cfg()
    s3 = client(
        's3',
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region,
        config=Config(
            s3={'addressing_style': 'path'},  # type: ignore[arg-type]
            connect_timeout=10,
            read_timeout=60,
            retries={'max_attempts': 3},  # type: ignore[arg-type]
        ),
    )
    return s3, cfg.bucket_name


def upload_file(local_path: str, key: str) -> None:
    s3, bucket = _get_s3()
    s3.upload_file(local_path, bucket, key)


def download_file(key: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3, bucket = _get_s3()
    s3.download_file(bucket, key, local_path)