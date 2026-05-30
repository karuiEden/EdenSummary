import logging
from contextlib import asynccontextmanager
from typing import Annotated, List

from fastapi import FastAPI, Header, HTTPException, Depends, UploadFile, Form

from system.config import get_app_cfg, get_llm_cfg, get_whisper_cfg, get_smtp_cfg, get_celery_cfg
from system.guards import equal_api_key, check_file, check_and_parse_emails, check_id
from system.job_store import create_job, get_status, get_result
from system.worker import process_job

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_app_cfg()
        get_llm_cfg()
        get_whisper_cfg()
        get_smtp_cfg()
        get_celery_cfg()
    except Exception as e:
        logger.critical(str(e))
        raise
    yield

app = FastAPI(lifespan=lifespan)

def check_api_key(api_key: Annotated[str | None, Header()]):
    if api_key == '' or not equal_api_key(str(api_key)):
        raise HTTPException(status_code=401)


@app.get("/health", dependencies=[Depends(check_api_key)])
def health():
    return {"status": "ok"}

@app.post("/v1/jobs", dependencies=[Depends(check_api_key)], status_code=202)
async def create_job_api(file: UploadFile, emails: Annotated[str | None, Form()]):
    if not await check_file(file):
        raise HTTPException(status_code=415)
    emails_list: List[str] = check_and_parse_emails(emails)
    resp = create_job(file, emails_list)
    if resp["status"] == "failed":
        raise HTTPException(status_code=507, detail=resp)
    process_job.delay(resp["job_id"])
    return resp

@app.get("/v1/jobs/{job_id}", dependencies=[Depends(check_api_key)])
def check_status_api(job_id: str):
    if not check_id(job_id):
        raise HTTPException(status_code=400)
    resp = get_status(job_id)
    if resp["status"] == "failed":
        raise HTTPException(status_code=404, detail=resp)
    return resp

@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(check_api_key)])
def get_result_api(job_id: str):
    if not check_id(job_id):
        raise HTTPException(status_code=400)
    resp = get_result(job_id)
    if resp["status"] != "done":
        raise HTTPException(status_code=409, detail=resp)
    return resp

