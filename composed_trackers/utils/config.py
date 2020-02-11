# Based on
# Copyright (c) Open-MMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py

# Changes:
#  Config:
#  - .update_dotted
#  - .setattr_dotted_name
#  - .to_flatten
#  - ._filename_ext
#  ConfigFlatten

import os.path as osp
from pathlib import Path
import sys
from argparse import ArgumentParser
from importlib import import_module

from addict import Dict

from .utils import collections_abc
from .utils import check_file_exist


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, k + '.')
        elif isinstance(v, collections_abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return parser


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    """

    @staticmethod
    def fromfile(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.endswith('.py'):
            module_name = osp.basename(filename)[:-3]
            if '.' in module_name:
                raise ValueError('Dots are not allowed in config file path.')
            config_dir = osp.dirname(filename)
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
        elif filename.endswith(('.yml', '.yaml', '.json')):
            import mmcv
            cfg_dict = mmcv.load(filename)
        else:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        return Config(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)
        """
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if filename:
            with open(filename, 'r') as f:
                super(Config, self).__setattr__('_text', f.read())
        else:
            super(Config, self).__setattr__('_text', '')

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    # Changes:
    ######################
    # additional methods
    ######################
    def update_dotted(self, dotted_names_dict, verbose=False):
        is_updated = False
        self._updates_list = []
        for name, value in dotted_names_dict.items():
            isu = self.setattr_dotted_name(self, name, value, verbose=verbose)
            is_updated = is_updated or isu

    def setattr_dotted_name(self, a, name, value, verbose=False):

        names = name.split('.')
        last_name = names[-1]
        n = len(names)
        for i in range(n):
            if i < n - 1:
                a = getattr(a, names[i])

        old_value = getattr(a, last_name, None)

        is_updated = old_value != value
        if is_updated:
            setattr(a, last_name, value)

        if verbose:
            if is_updated:
                s = f'{name:20}: {old_value:20} ---> {value}'
                self._updates_list.append(s)
                print(s)
        return is_updated

    # TODO:  Path().suffix
    @property
    def _filename_ext(self):
        return str(Path(self._filename)).split('.')[-1]

    def to_flatten(self, sep='.'):
        flatten = ConfigFlatten()
        flatten_add_args(flatten, self, sep=sep)
        return flatten


class ConfigFlatten():
    def __init__(self):
        self._dict = {}

    def add_item(self, key, value, **kwargs):
        self._dict[key] = value

    @property
    def __dict__(self):
        return self._dict


def flatten_add_args(flatten, cfg, prefix='', sep='.', excludes=[]):
    for k, v in cfg.items():
        if isinstance(v, str):
            flatten.add_item(prefix + k, v)
        elif isinstance(v, int):
            flatten.add_item(prefix + k, v, type=int)
        elif isinstance(v, float):
            flatten.add_item(prefix + k, v, type=float)
        elif isinstance(v, bool):
            flatten.add_item(prefix + k, v, action='store_true')
        elif isinstance(v, dict):
            flatten_add_args(flatten, v, k + sep, sep=sep)
        elif isinstance(v, collections_abc.Iterable):
            flatten.add_item(prefix + k, str(v))
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return flatten
