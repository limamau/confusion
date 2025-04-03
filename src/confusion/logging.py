import os
import time
from typing import Optional


class Logger:
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

    def log(self, message: str):
        if self.console:
            print(message, flush=True)

        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"{message}\n")

    def log_step(self, step: int, value: float, value_name: str, pre_str: str = ""):
        elapsed_time = time.time() - self.start_time
        self.log(
            f"{pre_str}Step: {step}, "
            + f"{value_name}: {value:.4e}, "
            + f"Elapsed time: {elapsed_time:.2f}s",
        )
