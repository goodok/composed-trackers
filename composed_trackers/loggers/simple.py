from pathlib import Path
import os
import pandas as pd
import warnings
from shutil import copyfile   # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python

# https://docs.neptune.ml/neptune-client/docs/experiment.html
# https://pytorch.org/docs/stable/tensorboard.html


from mmcv.fileio.io import dump

from ..utils.utils import is_notebook
from ..utils.streams.stdstream import StdOutStream, StdErrStream, FileWriter

from .base import BaseLogger


class SimpleLogger(BaseLogger):
    def __init__(self, name='name', description='Loggers', tags=[], debug=False,
                 root_path='./logbook',
                 exp_id=None,
                 exp_id_template='LOG-{AUTO}',
                 exp_id_template_debug = None,
                 verbose=0,
                 params={},             # Parameters of the experiment. After experiment creation params are read-only
                 properties={},         # Properties of the experiment. They are editable after experiment is created.
                 log_stdout=True,       # Not used when is_notebook==True
                 log_stderr=True,
                 #log_metrics=True,
                 **kwargs):
        self.name = name
        self.description = description
        self.tags = set(tags)
        self.root_path = root_path
        self.exp_id = exp_id
        self.exp_id_template = exp_id_template

        self.exp_id_template_debug = exp_id_template_debug
        if self.exp_id_template_debug is None:
            self.exp_id_template_debug = self.exp_id_template

        self.debug = debug
        self.verbose = verbose
        self.params = params
        self.properties = properties
        self.log_stdout = log_stdout
        self.log_stderr = log_stderr
        self._kwargs = kwargs
        self._stdout_stream = None
        self._stderr_stream = None
        #self.log_metrics = log_metrics
        self._metrics = {}

        self.root_path = Path(self.root_path)

        #self.initialize(**kwargs)

    def describe(self):
        print(self.__class__.__name__)
        print('    exp_id:', self.exp_id)
        print('      path:', self.path)

    def initialize(self, **kwargs):

        self.create_id()
        self.makedir()
        #self.intercept_std()
        self._dir_artifacts = self.path / 'artifacts'
        self.dump_params()
        self.dump_properties()
        self.dump_tags()

    def create_id(self):

        self.path = self.get_next_path()

    def intercept_std(self):
        self._stdout_stream = None
        self._stderr_stream = None

        if not is_notebook():
            if self.log_stdout:
                fn_log = self.path / 'stdout.txt'
                filewriter = FileWriter(fn_log)
                self._stdout_stream = StdOutStream([filewriter])
            if self.log_stderr:
                fn_log = self.path / 'stderr.txt'
                filewriter = FileWriter(fn_log)
                self._stdout_stream = StdErrStream([filewriter])


    def stop(self):
        print("BaseLogger stopping...", end=' ')
        if self._stdout_stream:
            self._stdout_stream.close()
        if self._stderr_stream:
            self._stderr_stream.close()
        self.dump_params()
        self.dump_properties()
        self.dump_tags()
        self.dump_metrics()
        print("Ok")


    def get_next_path(self):
        """
        Finds the next free path in an sequentially named list of files

        e.g. path_pattern = 'file-%s.txt':

        file-1.txt
        file-2.txt
        file-3.txt

        Runs in log(n) time where n is the number of existing files in sequence
        """
        i = 1

        # First do an exponential search
        while self.get_i_path(i).exists():
            i = i * 2

        # Result lies somewhere in the interval (i/2..i]
        # We call this interval (a..b] and narrow it down until a + 1 = b
        a, b = (i // 2, i)
        while a + 1 < b:
            c = (a + b) // 2 # interval midpoint
            a, b = (c, b) if self.get_i_path(c).exists() else (a, c)

        p = self.get_i_path(b)
        self.exp_id = p.name
        return p

    def get_i_path(self, i):
        templ = self.exp_id_template
        if self.debug:
            templ = self.exp_id_template_debug
        expid = templ.format(AUTO=i) 
        return self.root_path / expid

    def makedir(self, exist_ok=False):
        os.makedirs(self.path, exist_ok=exist_ok)

    def dump_params(self):
        dump(self.params, self.path / 'params.yaml')

    def dump_properties(self):
        dump(self.properties, self.path / 'properties.yaml')

    def dump_tags(self):
        dump(list(self.tags), self.path / 'tags.yaml')

    def dump_metrics(self):
        #if self.log_metrics:
        dump(self._metrics, self.path / 'metrics.json')

        try:
            df = self.metrics_to_df()
            df.to_csv(self.path / 'metrics.csv', index=False)
        except Exception as e:
            warnings.warn(f"Can't convert and save metrics to DataFrame. {e}")



    def log_artifact(self, artifact_filename, destination=None, local_only=False):
        # experiment.log_artifact('images/wrong_prediction_1.png')
        artifact_filename = Path(artifact_filename)
        if not artifact_filename.exists():
            warnings.warn(f'{artifact_filename} is not found')
            return
        if not artifact_filename.is_file():
            warnings.warn(f'{artifact_filename} is not a file')
            return

        if not self._dir_artifacts.exists():
            os.makedirs(self._dir_artifacts, exist_ok=True)

        if destination is None:
            destination = self._dir_artifacts / artifact_filename.name
        else:
            destination = Path(destination)
            target_directory = self._dir_artifacts / destination.parent
            os.makedirs(target_directory, exist_ok=True)
            destination = target_directory / destination.name

        if destination.exists():
            warnings.warn(f'destination {destination} is owerwriten.')
            
        copyfile(artifact_filename, destination)

    #send_artifact = log_artifact
    
    # log_text == log_and_send
    def save_and_log_artifact(self, value, filename='example.txt', local_only=False):
        """
        Save string as file and send it to remote.
        """
        destination = self._dir_artifacts / filename
        target_directory = destination.parent
        os.makedirs(target_directory, exist_ok=True)

        if destination.exists():
            warnings.warn(f'destination {destination} is owerwriten.')

        with open(destination, 'w') as f:
            f.write(value)

    #log_text = log_and_send

    def set_property(self, key, value):
        self.properties[key] = value
        self.dump_properties()

    def append_tag(self, tag, *tags):
        if isinstance(tag, list):
            tags_list = tag
        else:
            tags_list = [tag] + list(tags)
            
        self.tags = self.tags | set(tags_list)
        self.dump_tags()

    def log_metric(self, name, x, y=None):
        if self.verbose:
            print('BaseLogger: send_metric: ', name, x, y)

        #if self.log_metrics:
        if True:
            try:
                if name not in self._metrics:
                    self._metrics[name] = []
                d = self._metrics[name]
                if y is None:
                    y = x
                    x = -1
                x = int(x)
                d.append({'x': x, 'y': y})

                self.dump_metrics()

            except Exception as e:
                warnings.warn(f"Can't log metric '{name}': {e}")

    

    def metrics_to_df(self):
        #if not self.log_metrics:
        #    return
        metrics = self._metrics
        # epochs/batch dict
        x = {}
        for name in metrics.keys():
            for d in metrics[name]:
                i = d['x']
                if i not in x:
                    x[i] = {'i': i}
                x[i][name] = d['y']
        df = pd.DataFrame([row for i, row in x.items()])
        return df
    
