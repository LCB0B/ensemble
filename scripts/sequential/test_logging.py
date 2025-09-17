# %%
import random
import time
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class IntermittentWriter:
    """
    Wraps a real SummaryWriter and randomly raises OSError
    on add_scalar and flush() to simulate flaky storage.
    """

    def __init__(self, writer: SummaryWriter, fail_rate: float = 0.2) -> None:
        """
        Args:
            writer (SummaryWriter): The underlying writer to proxy.
            fail_rate (float): Probability [0.0,1.0] of raising OSError.
        """
        self._writer = writer
        self.fail_rate = fail_rate

    def add_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        """
        Forward to the real writer, or randomly fail.

        Args:
            tag (str): Metric name.
            scalar_value (float): Metric value.
            global_step (Optional[int]): Global step.
        """
        if random.random() < self.fail_rate:
            raise OSError("🔌 simulated add_scalar hiccup")
        self._writer.add_scalar(tag, scalar_value, global_step)

    def flush(self) -> None:
        """
        Forward to the real writer, or randomly fail on flush.

        Raises:
            OSError: If simulated failure occurs.
        """
        if random.random() < self.fail_rate:
            raise OSError("🔌 simulated flush hiccup")
        self._writer.flush()

    def close(self) -> None:
        """Forward close() so file handles get released."""
        self._writer.close()

    def __getattr__(self, name: str) -> Any:
        """Proxy any other calls/attributes to the real writer."""
        return getattr(self._writer, name)


import time
from typing import Any, Callable, Dict

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter


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


def main() -> None:
    """
    Interactive demo of RetryOrSkipTensorBoardLogger under intermittent I/O errors.
    """
    # 1) Instantiate your retry-wrapper with fewer retries & shorter delay for demo
    logger = RetryEverythingTensorBoardLogger(
        save_dir="logs_demo",
        name="demo",
        version="v0",
        max_retries=50,
        retry_delay=0.1,
        default_hp_metric=False,
        # flush_secs=0,
        # max_queue=1,
    )

    # first grab the real writer
    real = logger.experiment
    # wrap it in your intermittent failure simulator
    intermittent = IntermittentWriter(real, fail_rate=0.8)
    # then wrap *that* in the retrying proxy
    wrapped = RetryEverythingSummaryWriter(
        intermittent, max_retries=50, retry_delay=0.1
    )
    # finally replace
    logger._experiment = wrapped

    # 3) Call each method in turn; your retry logic will kick in on random OSErrors
    print("→ logging metrics...")
    logger.log_metrics({"accuracy": 0.42}, step=1)

    print("→ saving (flush)...")
    logger.save()

    print("→ finalizing...")
    logger.finalize("success")

    print("✅ All done (no uncaught exceptions)")  # if you see this, retries worked!


# %%
main()

# %%
