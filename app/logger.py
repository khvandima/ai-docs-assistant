import logging
import sys
from pathlib import Path
 
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
 
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
 
 
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
 
    if logger.handlers:
        return logger
 
    logger.setLevel(logging.INFO)
 
    # Все логи — app.log
    app_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
 
    # Только ошибки — errors.log
    error_handler = logging.FileHandler(LOG_DIR / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
 
    # Консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
 
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
 
    return logger
