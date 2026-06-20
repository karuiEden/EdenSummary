import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, List

from fastapi import FastAPI, Header, HTTPException, Depends, UploadFile, Form
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from eden_summary.core import get_x_api_key, get_storage_cfg, get_db_cfg, JobStatus, get_session, get_llm_cfg, get_whisper_cfg, get_smtp_cfg, get_celery_cfg, equal_api_key, check_file, check_and_parse_emails, check_id, create_job, get_status, get_result, record_result_edits
from eden_summary.worker import process_job


class ResultEdit(BaseModel):
    """Corrected structured summary submitted via PATCH /result. Send the full
    summary as returned by GET /result['structured']; unchanged fields are kept
    so they can serve as negative calibration labels (Q3)."""
    title: str = ''
    tldr: list = []
    decisions: list = []
    action_items: list = []
    risks: list = []


logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', "INFO").upper(), logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_x_api_key()
        get_llm_cfg()
        get_whisper_cfg()
        get_smtp_cfg()
        get_celery_cfg()
        get_db_cfg()
        get_storage_cfg()
    except Exception as e:
        logger.critical(str(e))
        raise
    yield

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

def check_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if x_api_key is None or x_api_key == '' or not equal_api_key(str(x_api_key)):
        raise HTTPException(status_code=401)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/jobs", dependencies=[Depends(check_api_key)], status_code=202)
async def create_job_api(file: UploadFile, emails: Annotated[str | None, Form()] = None, language: Annotated[str | None, Form()] = None, db_session: AsyncSession = Depends(get_session)):
    # Size check first: reject an oversized upload before check_file copies the whole
    # file to a temp for content sniffing.
    max_bytes = int(os.getenv('MAX_UPLOAD_MB', "500")) * 1024 * 1024
    if file.size and file.size > max_bytes:
        raise HTTPException(status_code=413)
    if not await check_file(file):
        raise HTTPException(status_code=415)
    emails_list: List[str] = check_and_parse_emails(emails)
    resp = await create_job(file, emails_list, db_session)
    if resp["status"] == JobStatus.FAILED:
        raise HTTPException(status_code=507, detail=resp)
    process_job.delay(resp["job_id"], language)
    return resp

@app.get("/v1/jobs/{job_id}", dependencies=[Depends(check_api_key)])
async def check_status_api(job_id: str, db_session: AsyncSession = Depends(get_session)):
    if not check_id(job_id):
        raise HTTPException(status_code=400)
    resp = await get_status(job_id, db_session)
    if resp["status"] == JobStatus.NON_EXISTING:
        raise HTTPException(status_code=404, detail=resp)
    return resp

@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(check_api_key)])
async def get_result_api(job_id: str, db_session: AsyncSession = Depends(get_session)):
    if not check_id(job_id):
        raise HTTPException(status_code=400)
    resp = await get_result(job_id, db_session)
    if resp["status"] == JobStatus.NON_EXISTING:
        raise HTTPException(status_code=404, detail=resp)
    if resp["status"] != "done":
        raise HTTPException(status_code=409, detail=resp)
    return resp

@app.patch("/v1/jobs/{job_id}/result", dependencies=[Depends(check_api_key)])
async def edit_result_api(job_id: str, edit: ResultEdit, db_session: AsyncSession = Depends(get_session)):
    if not check_id(job_id):
        raise HTTPException(status_code=400)
    resp = await record_result_edits(job_id, edit.model_dump(), db_session)
    if "error" in resp:
        # not found → 404; not DONE or no structured summary → 409 (mirrors GET)
        if resp.get("status") == JobStatus.NON_EXISTING:
            raise HTTPException(status_code=404, detail=resp)
        raise HTTPException(status_code=409, detail=resp)
    return resp

