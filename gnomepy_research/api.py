"""HTTP client for the gnome-controller research session API."""
from __future__ import annotations

import os

import requests

from gnomepy.auth import get_id_token
from gnomepy.config import config as gnome_config


def _api_base_url() -> str:
    return os.environ.get("GNOME_CONTROLLER_API_URL", gnome_config.CONTROLLER_API_URL).rstrip("/")


def _headers() -> dict[str, str]:
    return {
        "Authorization": get_id_token(),
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{_api_base_url()}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def create_session(
    session_name: str,
    spec_yaml: str = "",
    description: str = "",
    tags: list[str] | None = None,
    branch: str = "",
) -> dict:
    return _request("POST", "/research/sessions", json={
        "session_name": session_name,
        "spec_yaml": spec_yaml,
        "description": description,
        "tags": tags or [],
        "branch": branch or f"research/{session_name}",
    })


def get_session(session_name: str) -> dict:
    return _request("GET", f"/research/sessions/{session_name}")


def list_sessions(status: str | None = None, limit: int = 20) -> dict:
    params: dict = {"limit": str(limit)}
    if status:
        params["status"] = status
    return _request("GET", "/research/sessions", params=params)


def update_session(session_name: str, **fields) -> dict:
    return _request("PATCH", f"/research/sessions/{session_name}", json=fields)


def record_iteration(
    session_name: str,
    iteration: int,
    type: str,
    title: str,
    description: str,
    metrics: dict,
    metadata: dict,
    environment: dict,
    timestamp: str | None = None,
) -> dict:
    body: dict = {
        "iteration": iteration,
        "type": type,
        "title": title,
        "description": description,
        "metrics": metrics,
        "metadata": metadata,
        "environment": environment,
    }
    if timestamp:
        body["timestamp"] = timestamp
    return _request("POST", f"/research/sessions/{session_name}/iterations", json=body)


def add_note(session_name: str, content: str) -> dict:
    return _request("POST", f"/research/sessions/{session_name}/notes", json={"content": content})


def get_notes(session_name: str) -> list[dict]:
    return get_session(session_name).get("notes", [])
