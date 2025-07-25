"""
import logging
import sys

# --- Color Formatter (ANSI)
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG':    '\033[36m',   # Cyan
        'INFO':     '\033[32m',   # Green
        'WARNING':  '\033[33m',   # Yellow
        'ERROR':    '\033[31m',   # Red
        'CRITICAL': '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{msg}{self.RESET}"

# --- Basic formatters
plain_formatter = logging.Formatter(
    "%(levelname)-8s: %(name)s: %(message)s"
)

file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# --- Handlers
# 1. Colored terminal handler
color_handler = logging.StreamHandler(sys.stdout)
color_handler.setLevel(logging.DEBUG)
color_handler.setFormatter(ColorFormatter("%(levelname)-8s: %(name)s: %(message)s"))

# 2. Plain terminal handler (could go to stderr or a different stream)
plain_handler = logging.StreamHandler(sys.stderr)
plain_handler.setLevel(logging.DEBUG)
plain_handler.setFormatter(plain_formatter)

# 3. File handler (no color)
file_handler = logging.FileHandler("program.log", mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# --- Setup logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(color_handler)
logger.addHandler(plain_handler)
logger.addHandler(file_handler)

# --- Test logs
logger.debug("Debugging details")
logger.info("General info")
logger.warning("A warning")
logger.error("An error occurred")
logger.critical("Critical issue")
"""

import logging
import sys
from rich.logging import RichHandler

def setup_logger(name="wilson", level=logging.INFO, use_rich=True, log_to_file=None, no_color=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Already set up
        return logger

    # Formatter for plain output (no color)
    plain_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Terminal handler (colored or plain)
    if use_rich and not no_color:
        handler = RichHandler(markup=True, rich_tracebacks=True)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(plain_fmt)

    logger.addHandler(handler)

    # Optional file logging (plain format)
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file, mode='w')
        file_handler.setFormatter(plain_fmt)
        logger.addHandler(file_handler)

    return logger
