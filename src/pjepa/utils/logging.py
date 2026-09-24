import logging


try:
    from loguru import logger as logger
except Exception:
    logger = logging.getLogger("pjepa")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
