"""Monitoring utilities: Telegram alerts and a simple FastAPI dashboard."""
from __future__ import annotations

import requests
from fastapi import FastAPI

from utils.csv_utils import read_csv_safe


def send_telegram_alert(text: str, token: str, chat_id: str) -> None:
    """Send a message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})


app = FastAPI()


@app.get("/equity")
async def equity_curve() -> list[dict]:
    """Return equity log as a list of records for plotting."""
    df = read_csv_safe("equity_log.csv")
    return df.to_dict("records")
