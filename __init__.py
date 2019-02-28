
#__all__ = ['auxiliary', 'core', 'plotting', 'analysis']

import grimsel.grimsel as grimsel

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
