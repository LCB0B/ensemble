import getpass
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

try:
    from loguru import logger
except:
    print("Could not load loguru")
from pytorch_lightning.loggers import TensorBoardLogger


class RetryOrSkipTensorBoardLogger(TensorBoardLogger):
    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 5.0,
        max_skips: int = 5,
        **kwargs: Any,
    ):
        """
        A TensorBoard logger that retries on I/O errors and skips after repeated failures.

        Args:
            *args: Passed to TensorBoardLogger (root_dir, name, version, etc.).
            max_retries (int): How many times to retry on OSError.
            retry_delay (float): Seconds to wait between retries.
            max_skips (int): How many total skips allowed before raising.
            **kwargs: Forwarded to SummaryWriter.
        """
        super().__init__(*args, **kwargs)
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay
        self.max_skips: int = max_skips
        self._skips: int = 0

    def _reset_experiment(self) -> None:
        """
        Tear down the current writer so that `self.experiment` will
        re-create a fresh SummaryWriter on next access.
        """
        try:
            self.experiment.close()
        except Exception:
            pass
        self._experiment = None  # type: ignore

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics with retry on OSError (or ValueError caused by OSError)
        and skip logic.

        Args:
            metrics (Dict[str, float]): Metric names to values.
            step (int): Global step.
        """
        if self._skips >= self.max_skips:
            raise RuntimeError(
                f"Exceeded max skips ({self.max_skips}); aborting logging."
            )
        original_metrics, original_step = metrics, step

        for attempt in range(1, self.max_retries + 1):
            try:
                super().log_metrics(original_metrics, original_step)
                return
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # unwrap Lightning's ValueError wrapper
                cause = getattr(e, "__cause__", None)
                is_io = isinstance(e, OSError) or isinstance(cause, OSError)
                if not is_io:
                    # some other problem (e.g. wrong dtype) — propagate immediately
                    raise
                # it's an I/O failure: retry or skip
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] I/O failure {attempt}/{self.max_retries}, retrying in {self.retry_delay}s…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    self._skips += 1
                    print(
                        f"[TBLogger] Skipping log {self._skips}/{self.max_skips} after {self.max_retries} I/O failures."
                    )
                    self._reset_experiment()
                    return


class RetryTensorBoardLogger(TensorBoardLogger):
    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        """
        A TensorBoard logger that retries on I/O errors and skips after repeated failures.

        Args:
            *args: Passed to TensorBoardLogger (root_dir, name, version, etc.).
            max_retries (int): How many times to retry on OSError.
            retry_delay (float): Seconds to wait between retries.
            max_skips (int): How many total skips allowed before raising.
            **kwargs: Forwarded to SummaryWriter.
        """
        super().__init__(*args, **kwargs)
        self.max_retries: int = max_retries
        self.retry_delay: float = retry_delay

    def _reset_experiment(self) -> None:
        """
        Tear down the current writer so that `self.experiment` will
        re-create a fresh SummaryWriter on next access.
        """
        try:
            self.experiment.close()
        except Exception:
            pass
        self._experiment = None  # type: ignore

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics with retry on OSError (or ValueError caused by OSError)
        and skip logic.

        Args:
            metrics (Dict[str, float]): Metric names to values.
            step (int): Global step.
        """
        original_metrics, original_step = metrics, step

        for attempt in range(1, self.max_retries + 1):
            try:
                super().log_metrics(original_metrics, original_step)
                return
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # it's an I/O failure: retry or skip
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] Failure {attempt}/{self.max_retries}, retrying in {self.retry_delay}s…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    raise


import time
from typing import Any, Dict

from pytorch_lightning.loggers import TensorBoardLogger


class ExtendedRetryTensorBoardLogger(TensorBoardLogger):
    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        """
        A TensorBoard logger that retries on I/O errors and skips after repeated failures.

        Args:
            *args: Passed to TensorBoardLogger (root_dir, name, version, etc.).
            max_retries (int): How many times to retry on OSError.
            retry_delay (float): Seconds to wait between retries.
            **kwargs: Forwarded to SummaryWriter.
        """
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _reset_experiment(self) -> None:
        """
        Tear down the current writer so that `self.experiment` will
        re-create a fresh SummaryWriter on next access.
        """
        try:
            self.experiment.close()
        except Exception:
            pass
        self._experiment = None  # type: ignore

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log scalar metrics with retry on I/O errors.

        Args:
            metrics (Dict[str, float]): Metric names to values.
            step (int): Global step.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                super().log_metrics(metrics, step)
                return
            except KeyboardInterrupt:
                raise
            except Exception:
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] log_metrics failure {attempt}/{self.max_retries}, retrying…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    raise

    def save(self) -> None:
        """
        Override TensorBoardLogger.save to wrap its flush in retry logic.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                super().save()
                return
            except KeyboardInterrupt:
                raise
            except Exception:
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] save failure {attempt}/{self.max_retries}, retrying…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    raise

    def finalize(self, status: str) -> None:
        """
        Override TensorBoardLogger.finalize to wrap its flush in retry logic.

        Args:
            status (str): One of "success", "failed", etc.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                super().finalize(status)
                return
            except KeyboardInterrupt:
                raise
            except Exception:
                if attempt < self.max_retries:
                    print(
                        f"[TBLogger] finalize failure {attempt}/{self.max_retries}, retrying…"
                    )
                    self._reset_experiment()
                    time.sleep(self.retry_delay)
                else:
                    raise


class RetryEverythingSummaryWriter:
    """
    Proxy around SummaryWriter that retries every method call on Exception.

    Args:
        writer (SummaryWriter): underlying real writer.
        max_retries (int): how many attempts before giving up.
        retry_delay (float): seconds to wait between retries.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> None:
        self._writer = writer
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def __getattr__(self, name: str) -> Any:
        """
        Wrap any callable attribute of the real writer in retry logic.
        Non-callable attrs pass through unchanged.
        """
        orig = getattr(self._writer, name)
        if not callable(orig):
            return orig

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for i in range(1, self._max_retries + 1):
                try:
                    return orig(*args, **kwargs)
                except Exception as e:
                    if i < self._max_retries:
                        print(
                            f"[TBRetry] `{name}` failed {i}/{self._max_retries}: {e!r}, retrying…"
                        )
                        time.sleep(self._retry_delay)
                        continue
                    print(
                        f"[TBRetry] `{name}` giving up after {i}/{self._max_retries}: {e!r}"
                    )
                    return None

        return wrapped


class RetryEverythingTensorBoardLogger(TensorBoardLogger):
    """
    A TensorBoardLogger whose `.experiment` is wrapped to retry **all**
    SummaryWriter operations up to `max_retries` times.

    Args:
        *args: passed to TensorBoardLogger (save_dir, name, version, etc.).
        max_retries (int): how many retry attempts on any Exception.
        retry_delay (float): seconds between retries.
        **kwargs: forwarded to SummaryWriter.
    """

    def __init__(
        self,
        *args: Any,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # force instantiation of the real SummaryWriter
        real_writer: SummaryWriter = self.experiment
        # replace it with our retrying proxy
        self._experiment = RetryEverythingSummaryWriter(
            real_writer, max_retries=max_retries, retry_delay=retry_delay
        )


def configure_logger(log_file_path: Path) -> None:
    """
    Configures the logger by removing the default handler and adding file and console handlers.
    Sets up a global exception hook to log uncaught exceptions with their descriptions and prints
    the full traceback to the terminal.

    Args:
        log_file_path (Path): The full path to the log file.
    """

    # Remove the default handler.
    logger.remove()

    # Log to file with rotation and INFO level.
    logger.add(log_file_path, rotation="100 MB", level="INFO")

    # Log to console with INFO level.
    logger.add(sys.stdout, level="INFO")

    # Log the script start time and user
    user = getpass.getuser()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"Script started by user '{user}' at {start_time_str}")

    def handle_exception(
        exc_type: type, exc_value: BaseException, exc_traceback
    ) -> None:
        """
        Logs uncaught exceptions as errors with descriptions and prints the full traceback.

        Args:
            exc_type (type): The exception class.
            exc_value (BaseException): The exception instance.
            exc_traceback (traceback): The traceback object.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception occurred: {}", exc_value
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    # Set the global exception hook.
    sys.excepthook = handle_exception


def log_userwarnings_to_loguru() -> None:
    """
    Redirects all UserWarnings to loguru logger.
    """

    def custom_warning_handler(
        message, category, filename, lineno, file=None, line=None
    ):
        if issubclass(category, UserWarning):
            logger.warning(f"{category.__name__} at {filename}:{lineno} - {message}")

    warnings.showwarning = custom_warning_handler
