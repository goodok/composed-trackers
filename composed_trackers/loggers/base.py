import os
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import warnings
from shutil import copyfile   # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python

# https://docs.neptune.ml/neptune-client/docs/experiment.html
# https://pytorch.org/docs/stable/tensorboard.html


from mmcv.fileio.io import dump


# TODO:
# abstractmethod politics
# exp_id - abstarct

# aliases as methods
    


class BaseLogger():
    """Base class for loggers."""

    def __init__(self):
        pass
    
    #@abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
    
    def set_property(self, key, value):
        raise NotImplementedError

    
    def append_tag(self, tag, *tags):
        raise NotImplementedError
    
    def log_metric(self, name, x, y=None):
        raise NotImplementedError
    
    def log_artifact(self, filename, destination=None):
        raise NotImplementedError
    
    def log_text_as_artifact(self, text, destination=None, existed_temp_file=None):
        raise NotImplementedError
    
    
    # TODO: define as methods with self
    # aliases
    def send_metric(self, *args, **kwargs):
        self.log_metric( *args, **kwargs)

    # aliases
    def send_artifact(self, *args, **kwargs):
        self.log_artifact( *args, **kwargs)
