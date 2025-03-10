from typing import Optional
import traceback
import socket
from datetime import datetime

from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.callbacks import RichProgressBar, Callback
from src.utils.helper import alert, monitor, monitor_complete, yprint

__author__ = ' Yuhao Liu'


class SlackAlert(Callback):
    """
    Callback for sending a slack alert.
    """

    def __init__(self, exception_only: bool = False, disabled: bool = False,
                 ignore_keyboard_interrupt: bool = False, at_epoch: Optional[int] = None,
                 at_global_step: Optional[int] = None):
        """
        Args:
            exception_only (bool): Flag to indicate if the alert should only be sent on exceptions.
            disabled (bool): Flag to indicate if the callback is disabled.
            ignore_keyboard_interrupt (bool): Flag to indicate if keyboard interrupts should be ignored.
            at_epoch (int): Send slack alert at the end of the specified epoch.
            at_global_step (int): Send slack alert at the specified global step.
        """
        super().__init__()
        self.exception_only = exception_only
        self.pl_module_device = None
        self.exception_occurred = False
        self.disabled = disabled
        self.ignore_keyboard_interrupt = ignore_keyboard_interrupt
        self.at_epoch = at_epoch
        self.at_global_step = at_global_step
        self.hostname = socket.gethostname()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Send slack alert at the end of the specified epoch.
        """

        if self.at_epoch is not None and trainer.current_epoch >= self.at_epoch:
            title = f'Training progress: epoch {self.at_epoch} completed'
            self.at_epoch = None
        elif self.at_global_step is not None and trainer.global_step >= self.at_global_step:
            title = f'Training progress: global step {self.at_global_step} completed'
            self.at_global_step = None
        else:
            return
        wandb_run = trainer.logger.experiment
        wandb_url = wandb_run.url
        title += f' for run `{wandb_run.id}`'
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        device = str(trainer.strategy.root_device)
        message = (f'*{title}*\nTime completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}\n'
                   f'Wandb URL: {wandb_url}\n')
        alert(message)
        return

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
        if isinstance(exception, KeyboardInterrupt) and self.ignore_keyboard_interrupt:
                print('SlackAlert: Encountered keyboard interrupt. Slack alert not sent.\n'
                      'To enable slack alerts on keyboard interrupts, set `ignore_keyboard_interrupt` to False.')
                raise exception
        alert(message)
        self.exception_occurred = True
        raise exception

    @rank_zero_only
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Send slack alert on successful teardown.
        """
        if not self.exception_only and not self.exception_occurred and not self.disabled:
            title = f'{stage.capitalize()} completed'
            now = datetime.now() # current time
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            device = str(trainer.strategy.root_device) # cuda device
            message = f'*{title}*\n```Time completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}```'
            alert(message)
        return
