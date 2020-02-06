from pathlib import Path
import os
import pandas as pd
import warnings
import tempfile
import traceback
from shutil import copyfile   # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python

# https://docs.neptune.ml/neptune-client/docs/experiment.html
# https://pytorch.org/docs/stable/tensorboard.html


from mmcv.fileio.io import dump
from mmdet.utils.registry import Registry, build_from_cfg

from .loggers.simple import SimpleLogger
from .loggers.neptune import NeptuneLogger
from .loggers.base import BaseLogger

from .utils.log import print_color

LOGGERS = Registry('logger')
LOGGERS.register_module(SimpleLogger)
LOGGERS.register_module(NeptuneLogger)


class ComposedLoggers(BaseLogger):
    def __init__(self, name='name', description='Loggers', tags=[], params={}, debug=False, initialize_fn=None, **cfg):
        self.name = name
        self.description = description
        self.tags = tags
        self.params = params
        
        self.debug = debug
        
        self.cfg = cfg
        
        self._initialize_fn = initialize_fn

    def initialize(self):
        self.initialize_fn()

    def initialize_fn(self):
        if self._initialize_fn is not None:
            return self._initialize_fn()
        else:
            return self._initialize_fn_default()

    def _initialize_fn_default(self):
        """
        Default initialize function.
        """
        self.loggers = []
        
        default_keys = ['name', 'description', 'tags', 'params', 'debug']
        default_args = dict([(key, getattr(self, key)) for key in default_keys])
        
        suggested_id = None

        for logger_name in self.cfg['loggers']:
            
                cfg = default_args.copy()
                cfg['type'] = logger_name
                if suggested_id is not None:
                    cfg['exp_id'] = suggested_id
                cfg.update(self.cfg[logger_name])
                
                logger = build_from_cfg(cfg, LOGGERS)
                logger.initialize()
                if suggested_id is None:
                    suggested_id = logger.exp_id

                self.loggers.append(logger)
        pass

    def describe(self, ids_only=False):
        if not ids_only:
            keys = ['name', 'description', 'tags', 'debug']
            for key in keys:
                print(f'  {key:12}:', getattr(self, key, None))
        
            print('  loggers:', self.cfg['loggers'])
            for logger in self.loggers:
                print()
                logger.describe()
        else:
            for logger in self.loggers:
                print(f'{logger.__class__.__name__:15}:', end=' ')
                print_color(f'{logger.exp_id}', 'green')

    def stop(self):
        for logger in self.loggers:
            try:
                logger.stop()
            except:
                warnings.warn(f"Can't .stop for logger {logger}. {e}", UserWarning)
    
    def set_property(self, key, value):
        for logger in self.loggers:
            try:
                logger.set_property(key, value)
            except Exception as e:
                warnings.warn(f"Can't .set_property for logger {logger}. {e}", UserWarning)
    
    def append_tag(self, tag, *tags):
        for logger in self.loggers:
            try:
                logger.append_tag(tag, *tags)
            except Exception as e:
                warnings.warn(f"Can't .append_tag for logger {logger}. {e}", UserWarning)

    
    def log_metric(self, name, x, y=None):
        for logger in self.loggers:
            try:
                logger.log_metric(name, x, y)
            except Exception as e:
                warnings.warn(f"Can't .log_metric for logger {logger}. {e}", UserWarning)
                print(e)
                traceback.print_exc()

    
    def log_artifact(self, filename, destination=None):
        for logger in self.loggers:
            try:
                logger.log_artifact(filename, destination)
            except Exception as e:
                warnings.warn(f"Can't .log_artifact for logger {logger} {e}", UserWarning)
    
    def log_text_as_artifact(self, text, destination=None, existed_temp_file=None):
        fd = None
        try:
            fd, path = tempfile.mkstemp()
            
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(text)
            
            for logger in self.loggers:
                try:
                    logger.log_artifact(path, destination)
                except Exception as e:
                    warnings.warn(f"Can't .log_artifact for logger {logger}. {e}", UserWarning)
        except Exception as e:
            warnings.warn(f"Can't .log_text_as_artifact for composer {self} {e}", UserWarning)
        finally:
            if fd is not None:
                os.remove(path)
    
    @property
    def path(self):
        for logger in self.loggers:
            path = getattr(logger, 'path', None)
            if path is not None:
                return path
        
    
    # aliases
    save_and_log_artifact = log_text_as_artifact
