
import logging

def _get_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        f_handler = logging.StreamHandler()
        f_handler.setLevel(0)
        format_str = '> %(asctime)s - %(levelname)s - %(name)s - %(message)s'
        f_format = logging.Formatter(format_str, "%H:%M:%S")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

logger = _get_logger(__name__)

from grimsel.core.model_base import ModelBase
from grimsel.core.model_loop import ModelLoop

from grimsel.auxiliary.maps import Maps

from grimsel.auxiliary.multiproc import run_parallel, run_sequential

