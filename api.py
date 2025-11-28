"""Networking helpers for calling the chat completion endpoint."""

from __future__ import annotations

import time

import requests

from .config import API_KEY, BASE_URL, MODEL, RETRY_DELAY, RETRY_LIMIT, TIMEOUT


def safe_request(url, headers, payload, max_retries=RETRY_LIMIT, delay=RETRY_DELAY):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code != 200:
                print(f"⚠️ API error (status {resp.status_code}) on attempt {attempt}: {resp.text[:200]}")
                time.sleep(delay)
                continue
            try:
                return resp.json()
            except ValueError:
                print(f"⚠️ Invalid JSON response on attempt {attempt}: {resp.text[:200]}")
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error on attempt {attempt}: {e}")
            time.sleep(delay)
    print("❌ All retries failed.")
    return None


def chat_completion(messages, model=None):
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": (model or MODEL), "messages": messages}
    return safe_request(url, headers, payload)


def get_message_content(api_json) -> str:
    try:
        ch0 = api_json.get("choices", [{}])[0]
        msg = ch0.get("message", {})
        if isinstance(msg, dict) and msg.get("content"):
            return str(msg["content"]).strip()
        if ch0.get("text"):
            return str(ch0["text"]).strip()
    except Exception:
        pass
    return ""

