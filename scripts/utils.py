import os
import logging
from datetime import datetime

def get_logger(script_name):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{script_name}_{timestamp}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    # Only add handlers if none exist yet
    if not logger.handlers:
        # Console — clean, no timestamps
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(message)s"))

        # File — full detail with timestamps
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))

        logger.addHandler(ch)
        logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger