from __future__ import annotations

import re

FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    blocks = FENCED_BLOCK_RE.findall(text)
    if blocks:
        return max((block.strip() for block in blocks), key=len)
    return text.strip()
