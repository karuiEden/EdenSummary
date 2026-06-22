"""Request-validation guards used by the API: constant-time API-key comparison,
content-sniffing audio validation, recipient parsing, and UUID checks."""
import hmac
import shutil
import tempfile

from email_validator import validate_email, EmailNotValidError
from fastapi import UploadFile
from pydantic import UUID4

from .config import get_x_api_key
from eden_summary.transcribe import is_audio


def equal_api_key(in_api_key: str) -> bool:
    """Constant-time comparison of a supplied key against the configured one."""
    return hmac.compare_digest(in_api_key, get_x_api_key())

async def check_file(file: UploadFile) -> bool:
    """Validate that an upload is really audio/video: the libmagic MIME type must be
    audio/* or video/*, and ffprobe must confirm an audio stream. Rewinds the file
    afterwards so the caller can read it again."""
    import magic
    with tempfile.NamedTemporaryFile() as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        real_mime = magic.from_file(tmp.name, mime=True)
        real_check = is_audio(tmp.name)
    await file.seek(0)
    return (real_mime.startswith('audio/') or real_mime.startswith('video/')) and real_check

def check_and_parse_emails(emails_str: str | None):
    """Parse a comma-separated recipient string into a list of valid addresses,
    silently dropping invalid ones. Returns None when nothing valid remains."""
    if emails_str is None or emails_str == '':
        return None
    emails_list = list(map(str.strip, emails_str.split(',')))
    emails = []
    for email in emails_list:
        try:
            validate_email(email, check_deliverability=False)
            emails.append(email)
        except EmailNotValidError:
            pass
    if not emails:
        return None
    return emails

def check_id(job_id: str):
    """True if job_id is a valid UUID4 string."""
    try:
        UUID4(job_id)
        return True
    except ValueError:
        return False

