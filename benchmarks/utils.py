from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
from typing import TypeVar
from core.jinja_helper import process_template
parent_path: str = Path(__file__).parent.absolute().as_posix()

process_template = partial(process_template, parent_path=parent_path)

T = TypeVar("T", list[dict[str, Any]], dict[str, Any], str, Any)

