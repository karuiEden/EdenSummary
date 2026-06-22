"""SMTP delivery of the finished summary."""
import smtplib
from email.message import EmailMessage
from typing import List

from eden_summary.core import SMTPConfig, get_smtp_cfg


def send_email(recipients: List[str], subject: str, body: str):
    """Send the summary to all recipients over STARTTLS. A no-op when recipients is
    empty — a job may run without any address, and the result stays fetchable by id."""
    if not recipients:
        return
    config: SMTPConfig = get_smtp_cfg()
    with smtplib.SMTP(host=config.host, port=config.port) as server:
        server.ehlo()
        server.starttls()
        server.login(
            user=config.username,
            password=config.password
        )
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = config.sender
        msg['To'] = recipients
        server.send_message(msg)