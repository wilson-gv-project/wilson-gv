import os
from pathlib import Path
from importlib.resources import files

SUITE_ROOT = str(files('wilson_suite'))               # wilson_suite/
UTILS_ROOT = str(files('wilson_suite.wilson_utils'))  # wilson_suite/wilson_utils/


WORKFLOW_BASE_DIR = Path(os.environ.get("WORKFLOW_BASE_DIR", "../workflows")).expanduser().resolve()
# Path("~/workflows").expanduser() --  /home/vanda/workflows
# .resolve() -- converts the path to an absolute path; normalizes . and ..
