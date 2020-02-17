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
import collections
import argparse
from argparse import ArgumentParser
from importlib import import_module
import warnings
from addict import Dict

from .utils import collections_abc, check_file_exist, is_notebook


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
        warnings.warn('Use .create_arg_parser.')

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
    @property
    def _filename_ext(self):
        return Path(self._filename).suffix

    def to_flatten_dict(self, sep='.'):
        flatten = ConfigFlatten()
        flatten.load(source=self, sep=sep)
        return flatten

    def update_by_flatten(self, flatten_names_dict, verbose=False, sep='.'):
        is_updated = False
        self._updates_list = []
        for name, value in flatten_names_dict.items():
            isu = self.setattr_flatten(self, name, value, verbose=verbose, sep=sep)
            is_updated = is_updated or isu

        if verbose and len(self._updates_list) > 0:
            print('Updates of configuration:')
            for line in self._updates_list:
                print(line)

    def setattr_flatten(self, part, name, value, verbose=False, sep='.'):

        names = name.split(sep)
        last_name = names[-1]
        n = len(names)
        for i in range(n):
            if i < n - 1:
                part = getattr(part, names[i])

        old_value = getattr(part, last_name, None)

        is_updated = old_value != value
        if is_updated:
            setattr(part, last_name, value)

        if verbose:
            if is_updated:
                _name = f'{name} :'
                s = f'    {_name:20} {old_value:20} ---> {value}'
                self._updates_list.append(s)
        return is_updated

    def create_arg_parser(self, description=None, excludes=['_updates_list'], sep='.'):
        """
        Generate argparser from config file automatically.
        """
        parser = ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('config', help='config file path')
        self.add_args_to_parser(parser, source=self, sep=sep, excludes=excludes)
        return parser

    def add_args_to_parser(self, parser, source, prefix='', sep='.', excludes=[]):
        for k, v in source.items():
            if prefix + k in excludes:
                continue
            if isinstance(v, str):
                parser.add_argument('--' + prefix + k, default=v, help=' ')
            elif isinstance(v, bool):
                parser.add_argument('--' + prefix + k, type=bool, default=v, help=' ')
            elif isinstance(v, int):
                parser.add_argument('--' + prefix + k, type=int, default=v, help=' ')
            elif isinstance(v, float):
                parser.add_argument('--' + prefix + k, type=float, default=v, help=' ')
            elif isinstance(v, dict):
                # recursion
                group = parser.add_argument_group(prefix + k, '')
                self.add_args_to_parser(group, source=v, prefix=prefix + k + sep)
            elif isinstance(v, collections_abc.Iterable):
                parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+', default=v, help=' ')
            else:
                print(f"Can't parse key '{prefix + k}' of type '{type(v)}'")
        return parser


# Not used
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


class ConfigFlatten(collections.UserDict):
    def __init__(self):
        self.data = {}

    def load(self, source, sep='.', excludes=[]):
        self.add_part(source, sep=sep, excludes=excludes)

    def add_part(self, source, prefix='', sep='', excludes=[]):
        """
        Recursive
        """
        for k, v in source.items():
            if isinstance(v, str):
                self.add_item(prefix + k, v)
            elif isinstance(v, int):
                self.add_item(prefix + k, v, type=int)
            elif isinstance(v, float):
                self.add_item(prefix + k, v, type=float)
            elif isinstance(v, bool):
                self.add_item(prefix + k, v, action='store_true')
            elif isinstance(v, dict):
                # recursion
                self.add_part(source=v, prefix=prefix + k + sep, sep=sep)
            elif isinstance(v, collections_abc.Iterable):
                self.add_item(prefix + k, str(v))
            else:
                print(f"Can't parse key '{prefix + k}' of type '{type(v)}'")
        return self

    def add_item(self, key, value, **kwargs):
        self.data[key] = value


def get_shell_args(parser):
    args = parser.parse_args()
    shell_args = {}
    for k, v in args.__dict__.items():
        if v is not None:
            shell_args[k] = v
    return shell_args


def load_config_with_shell_updates(fn_config=None, notebook_shell_args={}, script_description=None, verbose=False, sep='.'):
    if is_notebook():
        shell_args = notebook_shell_args
        cfg = Config.fromfile(fn_config)
    else:
        fn_config = sys.argv[1]
        cfg = Config.fromfile(fn_config)
        parser = cfg.create_arg_parser(description=getattr(cfg.tracker, 'description', script_description), sep=sep)

        shell_args = get_shell_args(parser)
        _fn_config = shell_args.pop('config')
        assert _fn_config == fn_config

    cfg.update_by_flatten(shell_args, verbose=verbose, sep=sep)

    return cfg
