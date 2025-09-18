"""Monitoring utilities: Telegram alerts and a simple FastAPI dashboard."""
from __future__ import annotations

import requests
import pandas as pd
from fastapi import FastAPI


def send_telegram_alert(text: str, token: str, chat_id: str) -> None:
    """Send a message to a Telegram chat."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text})


app = FastAPI()


@app.get("/equity")
async def equity_curve() -> list[dict]:
    """Return equity log as a list of records for plotting."""
    df = pd.read_csv("equity_log.csv")
    return df.to_dict("records")
