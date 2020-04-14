import types
from functools import partial
from typing import Union, Optional, Dict, Iterable, Any, Callable, List, Tuple
from argparse import Namespace

import numpy as np


def adapt_to_pytorch_lightning(tracker):

    tracker.log_hyperparams = types.MethodType(Dummy.log_hyperparams, tracker)
    tracker.save = types.MethodType(Dummy.save, tracker)
    tracker.finalize = types.MethodType(Dummy.finalize, tracker)
    tracker.agg_and_log_metrics = types.MethodType(Dummy.agg_and_log_metrics, tracker)
    tracker._aggregate_metrics = types.MethodType(Dummy._aggregate_metrics, tracker)
    tracker._finalize_agg_metrics = types.MethodType(Dummy._finalize_agg_metrics, tracker)

    tracker._log_metrics = tracker.log_metrics
    tracker.log_metrics = types.MethodType(Dummy.log_metrics, tracker)

    addprop(tracker, 'version', Dummy._get_version)

    # from pytorch_lightning/loggers/base.py  init
    tracker._prev_step = -1
    tracker._metrics_to_agg: List[Dict[str, float]] = []
    tracker._agg_key_funcs = None
    tracker._agg_default_func = np.mean


def addprop(inst, name, method):
    cls = type(inst)
    if not hasattr(cls, '__perinstance'):
        cls = type(cls.__name__, (cls,), {})
        cls.__perinstance = True
        inst.__class__ = cls
    setattr(cls, name, property(method))


class Dummy(object):

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Nope: params (read-only) are saved when tracker was created.
        """
        return
    
        # For trackers params are read-only, so save them as properties
        # for k, v in vars(params).items():
        #    self.set_property(k, v)

        # TODO: HACK figure out where this is being set to true
        #self.experiment.debug = self.debug
        #params = self._convert_params(params)
        #params = self._flatten_dict(params)
        #self.experiment.argparse(Namespace(**params))

    def save(self):
        # Not used for trackers
        pass


    def finalize(self, status):
        # Not used for trackers
        pass


    def _get_version(self):
        # Not implemented
        return ''

    def log_metrics(self, metrics, step):
        # parameter 'step' is renamed, 
        self._log_metrics(metrics=metrics, index=step)


    # Below:
    # from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Aggregates and records metrics.
        This method doesn't log the passed metrics instantaneously, but instead
        it aggregates them and logs only if metrics are ready to be logged.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        agg_step, metrics_to_log = self._aggregate_metrics(metrics=metrics, step=step)

        if metrics_to_log is not None:
            print('agg_step:', agg_step)
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    def _aggregate_metrics(
            self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> Tuple[int, Optional[Dict[str, float]]]:
        """Aggregates metrics.
        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        Returns:
            sStep and aggregated metrics. The return value could be None. In such case, metrics
            are added to the aggregation list, but not aggregated yet.
        """
        # if you still receiving metric from the same step, just accumulate it
        if step == self._prev_step:
            self._metrics_to_agg.append(metrics)
            return step, None

        # compute the metrics
        agg_step, agg_mets = self._finalize_agg_metrics()

        # as new step received reset accumulator
        self._metrics_to_agg = [metrics]
        self._prev_step = step
        return agg_step, agg_mets

    def _finalize_agg_metrics(self):
        """Aggregate accumulated metrics. This shall be called in close."""
        # compute the metrics
        if not self._metrics_to_agg:
            agg_mets = None
        elif len(self._metrics_to_agg) == 1:
            agg_mets = self._metrics_to_agg[0]
        else:
            agg_mets = merge_dicts(self._metrics_to_agg, self._agg_key_funcs, self._agg_default_func)
        return self._prev_step, agg_mets
