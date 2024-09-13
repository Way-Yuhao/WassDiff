import traceback
from datetime import datetime
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.helper import alert, monitor, monitor_complete


class SlackAlert(Callback):
    """
    Callback for sending a slack alert.
    """

    def __init__(self, exception_only: bool = True):
        super().__init__()
        self.exception_only = exception_only  # Flag to indicate if the alert should only be sent on exceptions
        self.pl_module_device = None
        self.exception_occurred = False  # Flag to indicate if an exception occurred

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """
        Send slack alert on exceptions.
        """
        stack_trace = traceback.format_exc()
        # Prepare the alert message
        title = 'Exception Occurred:'
        message = f'*{title}*```{stack_trace}```'
        # Send the alert using your alert function
        alert(message)
        self.exception_occurred = True
        raise exception

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Send slack alert on successful teardown.
        """

        if not self.exception_only and not self.exception_occurred:
            title = f'{stage.capitalize()} completed'
            # Get the current time
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            # Get the device
            device = str(trainer.strategy.root_device)
            # Create the message
            message = f'*{title}*\n```Time completed: {formatted_time}\nDevice: {device}```'
            alert(message)
        return
