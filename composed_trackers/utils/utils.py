from __future__ import print_function
from __future__ import division

import sys
import subprocess
import humanize
import numpy as np
import pandas as pd
import re
import contextlib

import json
import os
from IPython.display import display, HTML
import warnings
from collections import OrderedDict
import platform


def is_notebook():
    try:
        # pylint: disable=pointless-statement,undefined-variable
        get_ipython
        return True
    except Exception:
        return False
    
    
    
def save_json(fname, d, pretty=False):
    fname = str(fname)
    with open(fname, 'w') as f:
        if pretty:
            json.dump(d, f, indent=4, sort_keys=True)
        else:
            json.dump(d, f)


def load_json(fname):
    fname = str(fname)
    with open(fname) as f:
        return json.load(f)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def wide_notebook(percents=70):
    display(HTML("<style>.container { width:70% !important; }</style>"))
    
    from .log import log_options
    
    log_options['max_name_length'] = 25
    log_options['max_shape_length'] = 20


# TODO: refactor: arguments - just list
def watermark(packages=['python', 'virtualenv', 'keras', 'tensorflow', 'nvidia', 'cudnn', 'hostname', 'torch', 'fastai', 'fastai_sparse'], return_string=True):
    lines = OrderedDict()
    if 'virtualenv' in packages:
        r = None
        if 'PS1' in os.environ:
            r = os.environ['PS1']
        elif 'VIRTUAL_ENV' in os.environ:
            r = os.environ['VIRTUAL_ENV']
        lines['virtualenv'] = r
    if 'python' in packages:
        r = sys.version.splitlines()[0]
        m = re.compile(r'([\d\.]+)').match(r)
        if m:
            r = m.groups()[0]
        lines['python'] = r
    if 'hostname' in packages:
        lines["hostname"] = platform.node()

    def find_in_lines(pip_list, package_name, remove_name=True):
        res = ''
        for line in pip_list:
            if hasattr(line, 'decode'):
                line = line.decode('utf-8')
            if package_name in line and line.startswith(package_name):
                if remove_name:
                    res = line.split(package_name)[1].strip()
                else:
                    res = line.strip()
                break
        return res

    if 'nvidia' in packages:
        lines['nvidia driver'] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
        try:
            r = subprocess.check_output(["nvcc", "--version"]).splitlines()
            r = find_in_lines(r, 'release', False)
            r = r.split('release')[1].strip()
            lines['nvidia cuda'] = r
        except:
            pass

    if ('cudnn' in packages) and sys.platform.startswith('linux'):
        try:
            with open('/usr/local/cuda/include/cudnn.h', 'r') as f:
                r = f.readlines()
            v1 = find_in_lines(r, '#define CUDNN_MAJOR')
            v2 = find_in_lines(r, '#define CUDNN_MINOR')
            v3 = find_in_lines(r, '#define CUDNN_PATCHLEVEL')
            lines['cudnn'] = "{}.{}.{}".format(v1, v2, v3)
        except:
            pass

    pip_list = subprocess.check_output(["pip", "list"]).splitlines()

    if 'keras' in packages:
        lines['keras'] = find_in_lines(pip_list, 'Keras')
        
    if 'tensorflow' in packages:
        lines['tensorflow-gpu'] = find_in_lines(pip_list, 'tensorflow-gpu')
        
    if 'torch' in packages:
        lines['torch'] = find_in_lines(pip_list, 'torch')


    if 'fastai' in packages:
        try:
            from fastai import version
            v = version.__version__
        except:
            v = None
        if v:
            lines['fastai'] = v

    if 'fastai_sparse' in packages:
        try:
            from fastai_sparse import version
            v = version.__version__
        except:
            v = None

        if v:
            lines['fastai_sparse'] = v

    parsed = ['python', 'virtualenv', 'keras', 'tensorflow', 'nvidia', 'cudnn', 'hostname', 'torch', 'fastai', 'fastai_sparse']
    for key in packages:
        if key not in parsed:
            lines[key] = find_in_lines(pip_list, key)

    res = ["{: <15} {}".format(k + ":", v) for (k, v) in lines.items()]

    s = "\n".join(res)
    print(s)

    if return_string:
        return s



def df_order_columns(df, columns_ordered=[]):
    """
    Order some columns, and remain other as was
    """
    columns = []
    for c in columns_ordered:
        if c in df.columns:
            columns.append(c)
    remains = [c for c in df.columns if c not in columns]
    columns = columns + remains

    assert len(columns) == len(df.columns)
    return df[columns]


def df_split_random(df, N, random_seed=None):
    """
    Split DataFrame on two parts.

    N : int
        Size of first part
    """
    random = np.random.RandomState(random_seed)

    all_local_indices = np.arange(len(df))
    shuffled = random.permutation(all_local_indices)

    df1 = df.iloc[shuffled[:N]]
    df2 = df.iloc[shuffled[N:]]
    return df1, df2

