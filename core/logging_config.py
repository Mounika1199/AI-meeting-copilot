import os
import datetime
import logging


def configure_logging():
    os.makedirs("logs", exist_ok=True)
    log_filename = f'logs/copilot_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(
        filename=log_filename,
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("\n\n--------Starting Meeting Copilot--------\n")
    logging.info(f"Logging initialized: {log_filename}")
