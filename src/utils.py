import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("ytl_poc")
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.propagate = False
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    base = setup_logging()
    return base if name is None else logging.getLogger(name)
