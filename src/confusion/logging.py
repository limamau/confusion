import logging
import os
import time
from abc import abstractmethod
from typing import Optional


class AbstractLogger:
    start_time: float

    @abstractmethod
    def log_msg(self, msg: str):
        pass

    def log_step(self, step: int, value: float, value_name: str, pre_str: str = ""):
        elapsed_time = time.time() - self.start_time
        self.log_msg(
            f"{pre_str}Step: {step}, "
            + f"{value_name}: {value:.4e}, "
            + f"Elapsed time: {elapsed_time:.2f}s"
        )


class PrintOnlyLogger(AbstractLogger):
    def __init__(self):
        self.start_time = time.time()

    def log_msg(self, msg: str):
        print(msg, flush=True)

    def log_step(self, step: int, value: float, value_name: str, pre_str: str = ""):
        elapsed_time = time.time() - self.start_time
        self.log_msg(
            f"{pre_str}Step: {step}, "
            + f"{value_name}: {value:.4e}, "
            + f"Elapsed time: {elapsed_time:.2f}s"
        )


class RootLogger(AbstractLogger):
    def __init__(self, log_file: str):
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.addHandler(logging.FileHandler(log_file))

        self.log_msg("RootLogger initialized.")

    def log_msg(self, msg: str):
        self.logger.info(msg)


class StandardLogger(AbstractLogger):
    def __init__(
        self, log_file: Optional[str] = None, console: bool = True, erase=False
    ):
        self.log_file = log_file
        self.console = console
        self.start_time = time.time()

        if log_file is not None:
            if erase:
                if os.path.exists(log_file):
                    os.remove(log_file)
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_msg(self, msg: str):
        if self.console:
            print(msg, flush=True)

        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"{msg}\n")
