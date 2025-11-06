from loguru import logger as logger
import sys
import time

logger.remove()

logger.add(
    sink=sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)


def log_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper
