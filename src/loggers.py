import time
from pytorch_lightning.loggers import TensorBoardLogger

class RetryTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, max_retries: int = 5, retry_delay: float = 0.5, **kwargs):
        """
        A TensorBoard logger that retries logging on failure.

        Args:
            max_retries (int): Maximum number of retry attempts.
            retry_delay (float): Time in seconds to wait between retries.
        """
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def log_metrics(self, metrics: dict, step: int) -> None:
        """
        Logs metrics with retry on failure.

        Args:
            metrics (dict): Metrics to log.
            step (int): Training step.
        """
        for attempt in range(self.max_retries):
            try:
                super().log_metrics(metrics, step)
                break  # Exit loop if successful
            except OSError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay + 0
                    print(
                        f"Logging failed on attempt {attempt + 1}/{self.max_retries} due to {e}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Logging failed on the final attempt {self.max_retries}/{self.max_retries}. "
                        "Raising exception."
                    )
                    raise e
