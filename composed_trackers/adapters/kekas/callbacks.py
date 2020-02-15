"A `Callback` that saves tracked metrics into a Neptune"
#Contribution from devforfu: https://nbviewer.jupyter.org/gist/devforfu/ea0b3fcfe194dad323c3762492b05cae

from collections import defaultdict
import numpy as np
import time

from kekas.callbacks import Callback
from kekas.utils import DotDict, get_opt_lr

__all__ = ['MetricsMonitor']


class MetricsMonitor(Callback):
    "A `Callback` that send history of metrics to `neptune`."


    def __init__(self, logger, simulation=False): 
        self.logger = logger
        self.total_iter = 0
        self.train_iter = 0
        self.val_iter = 0
        self.train_batch_iter = 0
        self.val_batch_iter = 0
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.epochs_total = 1
        self.simulation = simulation

    def update_total_iter(self, mode: str) -> None:
        if mode == "train":
            self.train_iter += 1
            self.train_batch_iter += 1
        if mode == "val":
            self.val_iter += 1
            self.val_batch_iter +=1
            self.total_iter += 1 #?????

    def on_train_begin(self, state: DotDict) -> None:
        self.train_iter = 0
        self.val_iter = 0

        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict):
        self.train_batch_iter = 0
        self.val_batch_iter = 0
        self.val_t0 = time.time()
        self.train_t0 = time.time()


    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            for name, metric in state.core.batch_metrics["train"].items():
                self.train_metrics[name].append(float(metric))

#                lr = get_opt_lr(state.core.opt)
#                self.train_writer.add_scalar("batch/lr",
#                                             float(lr),
#                                             global_step=self.train_iter)

            self.update_total_iter(state.core.mode)

        elif state.core.mode == "val":
            for name, metric in state.core.batch_metrics["val"].items():
                self.val_metrics[name].append(float(metric))

            self.update_total_iter(state.core.mode)


    def on_epoch_end(self, epoch: int, state: DotDict) -> None:

        #print("\n  MetricsMonitor: \n  state.core.epoch_metrics:", state.core.epoch_metrics)

        if state.core.mode == "train":
            # cacl mean by batch
            for name, metric in self.train_metrics.items():
                mean = np.mean(metric[-self.train_batch_iter:])
                self.send_metric(name, self.epochs_total, mean)

            # other metrics
            for mode, metrics in state.core.epoch_metrics.items():
                for name, metric in metrics.items():
                    if name != 'loss':
                        self.send_metric(mode + "_" + name, self.epochs_total, metric)

            self.send_metric('epoch_time_train', self.epochs_total, time.time() - self.train_t0)

        if state.core.mode == "val":
            # cacl mean by batch
            for name, metric in self.val_metrics.items():
                mean = np.mean(metric[-self.val_batch_iter:])  # last epochs vals
                self.send_metric("val_" + name, self.epochs_total, mean)

            # other metrics
            for mode, metrics in state.core.epoch_metrics.items():
                for name, metric in metrics.items():
                    if name != 'loss':
                        self.send_metric(mode + "_" + name, self.epochs_total, metric)

            lr = get_opt_lr(state.core.opt)
            self.send_metric("LR", self.epochs_total, lr)

            self.send_metric('epoch_time_valid', self.epochs_total, time.time() - self.val_t0)

            self.epochs_total +=1

    def send_metric(self, name, x, y):
        if not self.simulation and (self.logger is not None):
            self.logger.send_metric(name, x, y)
        else:
            print('\n', name, x, y)
