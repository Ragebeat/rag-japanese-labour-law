"""Database connection settings, read the same way by every script.

.env in the project root is loaded without overriding variables that are already set.
The IDE used to load .env; from a bare terminal scripts failed with
"fe_sendauth: no password supplied", and depending on where a script is launched
from is no basis for reproducible measurements. No python-dotenv: one dependency
for ten lines is not worth it.
"""

import os
import pathlib


def load_env():
    path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


load_env()

DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://rag:{}@localhost:5433/ragdb".format(
        os.environ.get("POSTGRES_PASSWORD", "")
    ),
)
