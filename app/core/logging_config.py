import logging
import os
from pathlib import Path

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"

LOG_DIR = Path("app/logs")
LOG_FILE = LOG_DIR / "app.log"

def setup_logging():
    handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        )
    )
    handlers.append(console_handler)

    if LOG_TO_FILE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=LOG_LEVEL,
        handlers=handlers,
        force=True,
    )
