import smtplib
from email.message import EmailMessage
from typing import List

from system.config import SMTPConfig, get_smtp_cfg


def send_email(recipients: List[str], subject: str, body: str):
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