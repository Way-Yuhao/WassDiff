import os
from typing import Optional
import traceback
import socket
import requests
from dotenv import load_dotenv
from datetime import datetime
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.callbacks import Callback
from src.utils import RankedLogger

__author__ = ' Yuhao Liu'

logger = RankedLogger(name=__name__, rank_zero_only=True)


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
        self.webhook_url = None
        self.configured = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        path_to_restore = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        load_dotenv('.env')
        os.chdir(path_to_restore)
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if self.disabled:
            logger.debug('SlackAlert disabled. No alerts will be sent to Slack.')
            self.configured = False
            return
        if self.webhook_url is None:
            logger.warning('SlackAlert not configured. To send alerts to slack, '
                           'set SLACK_WEBHOOK_URL in .env file under project root directory.')
            self.configured = False
        else:
            logger.info('SlackAlert configured. Monitoring alerts to be relayed to Slack..')
            self.configured = True
        return

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Send slack alert at the end of the specified epoch.
        """
        if self.disabled:
            return
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
        self.alert_(message)
        return

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """
        Send slack alert on exceptions.
        """
        if self.disabled:
            return
        stack_trace = traceback.format_exc()
        device = str(trainer.strategy.root_device)
        now = datetime.now().replace(microsecond=0)
        # Prepare the alert message
        title = f'Exception Occurred'
        message = f'*{title}*```{stack_trace}```\nHost: {self.hostname}\nDevice: {device}\nTime: {now}'
        # Send the alert using your alert function
        if isinstance(exception, KeyboardInterrupt) and self.ignore_keyboard_interrupt:
                logger.debug('SlackAlert: Encountered keyboard interrupt. Slack alert not sent.\n'
                      'To enable slack alerts on keyboard interrupts, set `ignore_keyboard_interrupt` to False.')
                # raise exception
                return
        self.alert_(message)
        self.exception_occurred = True
        # raise exception
        return

    @rank_zero_only
    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """
        Send slack alert on successful teardown.
        """
        if not self.exception_only and not self.exception_occurred and self.configured:
            title = f'{stage.capitalize()} completed'
            now = datetime.now() # current time
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            device = str(trainer.strategy.root_device) # cuda device
            message = f'*{title}*\n```Time completed: {formatted_time}\nHostname: {self.hostname}\nDevice: {device}```'
            self.alert_(message)
        return

    def alert_(self, message=str) -> None:
        """
        Sends a message to a designated slack channel, which a SLACK_WEBHOOK_URL to be set in .env file.
        If webhook URL is not found, the message is printed to stdout in red.
        :param message:
        :return:
        """
        if not self.configured:
            logger.debug('SlackAlert not configured. No alerts sent.')
            return
        data = {'text': message,
                'username': 'Webhook Alert',
                'icon_emoji': ':robot_face:'}
        response = requests.post(self.webhook_url, json=data)
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )
        return