from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_first_json_block(text: str) -> str:
    fence = "```json"
    start = text.find(fence)
    if start < 0:
        raise ValueError("No ```json fenced block found")
    start = start + len(fence)
    end = text.find("```", start)
    if end < 0:
        raise ValueError("Unclosed ```json fenced block")
    return text[start:end].strip()


def _resolve_path(project_root: Path, value: Any) -> Any:
    if isinstance(value, str):
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((project_root / p).resolve())
    if isinstance(value, list):
        return [_resolve_path(project_root, v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_path(project_root, v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class BaocrPaths:
    raw: Dict[str, Any]
    project_root: Path

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def get_path(self, key: str, default: Optional[str] = None) -> Optional[str]:
        v = self.raw.get(key)
        if v is None:
            return default
        return str(v)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


def load_paths_from_md(md_path: str | Path) -> BaocrPaths:
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
    json_str = _extract_first_json_block(text)
    raw = json.loads(json_str)

    pr = raw.get("project_root")
    project_root = Path(pr).resolve() if pr else md_path.resolve().parents[0]
    resolved = _resolve_path(project_root, raw)
    return BaocrPaths(raw=resolved, project_root=project_root)
