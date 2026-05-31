from pathlib import Path

from boto3 import client

from eden_summary.core import get_storage_cfg

cfg = get_storage_cfg()

s3_client = client(
    's3',
    endpoint_url=cfg.endpoint,
    aws_access_key_id=cfg.access_key,
    aws_secret_access_key=cfg.secret_key,
    region_name=cfg.region
)

def upload_file(local_path: str, key: str):
    s3_client.upload_file(local_path, cfg.bucket_name, key)

def download_file(key: str, local_path: str):
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(cfg.bucket_name, key, local_path)