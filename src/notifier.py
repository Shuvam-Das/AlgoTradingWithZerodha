import smtplib
from email.message import EmailMessage
import logging
import os
import secrets
try:
    import subprocess  # type: ignore  # nosec
except Exception:
    subprocess = None  # type: ignore
from typing import Optional, Callable, Dict
import time
import random

_last_call: Dict[str, float] = {}


def retry(max_attempts: int = 3, base_delay: float = 0.5, backoff: float = 2.0, jitter: float = 0.1):
    def deco(func: Callable):
        def wrapped(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    sleep = base_delay * (backoff ** (attempt - 1))
                    # jitter: use secrets for unpredictability (not crypto-critical here)
                    jitter_val = (secrets.randbelow(1000) / 1000.0) * jitter
                    # random direction
                    if secrets.randbelow(2) == 0:
                        sleep = sleep * (1 - jitter_val)
                    else:
                        sleep = sleep * (1 + jitter_val)
                    time.sleep(sleep)
        return wrapped
    return deco


def rate_limit(min_interval: float = 1.0):
    """Simple per-function rate limiter (min seconds between calls)."""
    def deco(func: Callable):
        key = func.__name__
        def wrapped(*args, **kwargs):
            now = time.time()
            last = _last_call.get(key, 0)
            if now - last < min_interval:
                logging.getLogger('atwz.notify').debug('Rate limit: skipping %s', key)
                return None
            _last_call[key] = now
            return func(*args, **kwargs)
        return wrapped
    return deco


logger = logging.getLogger("atwz.notify")


def send_email(smtp_server: str, smtp_port: int, username: str, password: str, to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg['From'] = username
    msg['To'] = to
    msg['Subject'] = subject
    msg.set_content(body)
    with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as s:
        s.starttls()
        s.login(username, password)
        s.send_message(msg)
    logger.info(f"Email sent to {to}: {subject}")


def desktop_alert(message: str) -> None:
    try:
        if os.name == 'posix':
            # ensure notify-send exists
            from shutil import which
            if subprocess is not None and which('notify-send'):
                # notify-send is a constant trusted binary; mark as nosec for bandit
                subprocess.run(["notify-send", "AlgoBot", message])  # nosec
            else:
                logger.debug('notify-send not found; skipping desktop alert')
        else:
            print("ALERT:", message)
    except Exception as e:
        logger.warning("desktop alert failed: %s", e)


def notify_telegram(bot_token: str, chat_id: str, message: str) -> None:
    """Send a Telegram message via bot API. This is a simple helper; network errors will raise."""
    import requests  # type: ignore[import]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    resp.raise_for_status()
    logger.info("Sent telegram message to %s", chat_id)


def _notify_telegram_from_env_impl(message: str) -> None:
    bot = os.getenv('TELEGRAM_BOT_TOKEN')
    chat = os.getenv('TELEGRAM_CHAT_ID')
    if not bot or not chat:
        logger.debug('Telegram creds not configured in env')
        return
    try:
        notify_telegram(bot, chat, message)
    except Exception:
        logger.exception('Failed to send telegram notification')


# Apply retry and rate limiting to env-based notifier
notify_telegram_from_env = rate_limit(1.0)(retry(max_attempts=3)(_notify_telegram_from_env_impl))

