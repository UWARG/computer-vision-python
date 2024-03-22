import datetime
import logging
import queue

from utilities.workers import queue_proxy_wrapper
from utilities.workers import worker_controller

def logging_worker(input_queue: queue_proxy_wrapper.QueueProxyWrapper,
                   controller: worker_controller.WorkerController):
    logger = logging.getLogger('airside_logger')

    filename = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
    file_handler = logging.FileHandler(filename=filename, mode="w")    # Handles logging to file
    stream_handler = logging.StreamHandler()                           # Handles logging to terminal

    formatter = logging.Formatter(
        fmt='%(asctime)s: [%(levelname)s] %(message)s',
        datefmt='%I:%M:%S',
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    while not controller.is_exit_requested():
        controller.check_pause()
    
        logging_message, level = input_queue.queue.get()

        if logging_message is None:
            break

        if level == logging.DEBUG:
            logger.debug(f"{logging_message}")
        elif level == logging.INFO:
            logger.info(f"{logging_message}")
        elif level == logging.WARNING:
            logger.warning(f"{logging_message}")
        elif level == logging.ERROR:
            logger.error(f"{logging_message}")
        elif level == logging.CRITICAL:
            logger.critical(f"{logging_message}")
