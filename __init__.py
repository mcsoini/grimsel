
#__all__ = ['auxiliary', 'core', 'plotting', 'analysis']

import logging

def _get_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        f_handler = logging.StreamHandler()
        f_handler.setLevel(0)
        format_str = '--> %(levelname)s - %(asctime)s - %(name)s - %(message)s'
        f_format = logging.Formatter(format_str)
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger
