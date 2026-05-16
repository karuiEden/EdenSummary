from pathlib import Path

from email_validator import validate_email, EmailNotValidError
from fastapi import UploadFile

from config import X_API_KEY
from system.config import audio_formats


def equal_api_key(in_api_key: str):
    return in_api_key == X_API_KEY

def check_file(file: UploadFile):
    file_ext = Path(str(file.filename)).suffix.lower()
    return file_ext in audio_formats.keys() and file.content_type in audio_formats.values() and audio_formats[file_ext] == file.content_type

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