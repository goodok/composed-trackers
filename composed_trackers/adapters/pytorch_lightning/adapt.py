import types
from functools import partial
from typing import Union, Optional, Dict, Iterable, Any, Callable, List
from argparse import Namespace


def adapt_to_pytorch_lightning(tracker):

    tracker.log_hyperparams = types.MethodType(Dummy.log_hyperparams, tracker)
    tracker.save = types.MethodType(Dummy.save, tracker)


    tracker._log_metrics = tracker.log_metrics
    tracker.log_metrics = types.MethodType(Dummy.log_metrics, tracker)

    addprop(tracker, 'version', Dummy._get_version)


def addprop(inst, name, method):
    cls = type(inst)
    if not hasattr(cls, '__perinstance'):
        cls = type(cls.__name__, (cls,), {})
        cls.__perinstance = True
        inst.__class__ = cls
    setattr(cls, name, property(method))


class Dummy(object):

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # For trackers params are read-only, so save them as properties
        for k, v in vars(params).items():
            self.set_property(k, v)

        # TODO: HACK figure out where this is being set to true
        #self.experiment.debug = self.debug
        #params = self._convert_params(params)
        #params = self._flatten_dict(params)
        #self.experiment.argparse(Namespace(**params))

    def save(self):
        # Not used for trackers
        pass

    def _get_version(self):
        # Not implemented
        return ''

    def log_metrics(self, scalar_metrics, step):
        # parameter 'step' is renamed
        self._log_metrics(scalar_metrics, index=step)
