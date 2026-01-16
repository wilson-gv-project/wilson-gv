import os
from pathlib import Path

UTILS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE_ROOT = os.path.abspath(os.path.join(UTILS_ROOT, '..'))


WORKFLOW_BASE_DIR = Path(os.environ.get("WORKFLOW_BASE_DIR", "../workflows")).expanduser().resolve()
# Path("~/workflows").expanduser() --  /home/vanda/workflows
# .resolve() -- converts the path to an absolute path; normalizes . and ..
