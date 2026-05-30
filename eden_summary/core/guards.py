import hmac
from pathlib import Path

import magic
from email_validator import validate_email, EmailNotValidError
from fastapi import UploadFile
from pydantic import UUID4

from eden_summary.core.config import audio_formats, X_API_KEY


def equal_api_key(in_api_key: str):
    return hmac.compare_digest(in_api_key, X_API_KEY)

async def check_file(file: UploadFile):
    file_ext = Path(str(file.filename)).suffix.lower()
    real_mime = magic.from_descriptor(file.file.fileno(), mime=True)
    await file.seek(0)
    return (file_ext in audio_formats.keys() and file.content_type in audio_formats.values()
            and audio_formats[file_ext] == file.content_type) and real_mime == audio_formats[file_ext]

def check_and_parse_emails(emails_str: str | None):
    if emails_str is None or emails_str == '':
        return None
    emails_list = list(map(str.strip, emails_str.split(',')))
    emails = []
    for email in emails_list:
        try:
            validate_email(email)
            emails.append(email)
        except EmailNotValidError:
            pass
    if not emails:
        return None
    return emails

def check_id(job_id: str):
    try:
        UUID4(job_id)
        return True
    except ValueError:
        return False

