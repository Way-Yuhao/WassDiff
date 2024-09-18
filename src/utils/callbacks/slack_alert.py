import traceback
import socket
from datetime import datetime
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.helper import alert, monitor, monitor_complete


class SlackAlert(Callback):
    """
    Callback for sending a slack alert.
    """

    def __init__(self, exception_only: bool = False, disabled: bool = False):
        super().__init__()
        self.exception_only = exception_only  # Flag to indicate if the alert should only be sent on exceptions
        self.pl_module_device = None
        self.exception_occurred = False  # Flag to indicate if an exception occurred
        self.disabled = disabled  # Flag to indicate if the callback is disabled

        self.hostname = socket.gethostname()

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """
        Send slack alert on exceptions.
        """
        if self.disabled:
            raise exception
        stack_trace = traceback.format_exc()
        device = str(trainer.strategy.root_device)
        now = datetime.now().replace(microsecond=0)
        # Prepare the alert message
        title = f'Exception Occurred'
        message = f'*{title}*```{stack_trace}```\nHost: {self.hostname}\nDevice: {device}\nTime: {now}'
        # Send the alert using your alert function
        alert(message)
        self.exception_occurred = True
        raise exception

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Send slack alert on successful teardown.
        """

        if not self.exception_only and not self.exception_occurred and not self.disabled:
            title = f'{stage.capitalize()} completed'
            # Get the current time
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            # Get the device
            device = str(trainer.strategy.root_device)
            # Create the message
            message = f'*{title}*\n```Time completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}```'
            alert(message)
        return
