from pathlib import Path
import os
import warnings
from shutil import copyfile   # https://stackoverflow.com/questions/123198/how-do-i-copy-a-file-in-python

from mmcv.fileio.io import dump
import neptune as neptune
from neptune.internal.streams.channel_writer import ChannelWriter
from neptune.internal.channels.channels import ChannelNamespace

from .base import BaseLogger

class NeptuneLogger(BaseLogger):
    
    def __init__(self, name='name', description='Loggers', tags=[], debug=False, 
                 params={},
                 properties={},
                 project=None,
                 fn_token=None,
                 exp_id=None,
                 upload_stdout=False,
                 upload_stderr=False,
                 **kwargs):
        
        self.name = name
        self.description = description
        self.tags = set(tags)
        self.debug = debug
        self.params = params
        self.properties = properties
        
        self.project = project
        self.fn_token = fn_token
        
        self.exp_id = exp_id
        
        kwargs['upload_stdout'] = upload_stdout
        kwargs['upload_stderr'] = upload_stderr
        
        self._kwargs = kwargs


    def describe(self):
        print(self.__class__.__name__)
        print('    exp_id:', self.exp_id)
        print('   project:', self.project)

    def initialize(self):
        
        
        if self.fn_token is not None:
            with open(os.path.expanduser(self.fn_token), 'r') as f:
                token = f.readline().splitlines()[0]
                os.environ['NEPTUNE_API_TOKEN'] = token

        if self.debug:
            neptune.init(project_qualified_name='dry-run/project',
                         backend=neptune.OfflineBackend())
        else:
            neptune.init(api_token=self.api_key,
                         project_qualified_name=self.project)
        
        
        self.internal = neptune.create_experiment(name=self.name,
                                                         params=self.params,
                                                         properties=self.properties,
                                                         tags=self.tags,
                                                         #upload_source_files=self.upload_source_files,
                                                         **self._kwargs)
        
        self.exp_id = self.internal.id
        
    
    def initialize_old(self, neptune={}, **kwargs):
        neptune_cfg = neptune
        #print('neptune_cfg:', neptune_cfg)
        if not self.test:
            with open(os.path.expanduser(neptune_cfg.fn_token), 'r') as f:
                token = f.readline().splitlines()[0]
            os.environ['NEPTUNE_API_TOKEN'] = token
            # TODO:use OfflineBackend ???
            neptune_client.init(project_qualified_name=neptune_cfg.project)
            
            # create experiment in the project defined above
            exp = neptune_client.create_experiment(name=self.name,
                                    description=self.description,
                                    params=self.params,
                                    properties=self.properties, 
                                    tags=list(self.tags),
                                    upload_stdout=self.log_stdout,
                                    upload_stderr=self.log_stderr,
                                    )
            self.exp = exp
            self.id = exp.id
            self.path = self.root_path / self.id
            self.makedir()
            self.intercept_std()
            self._dir_artifacts = self.path / 'artifacts'
            self.dump_params()
            self.dump_properties()
            self.dump_tags()

        else:
            self.exp = None
            super().initialize()
            #self.create_id()
            #self.makedir()
            #self.intercept_std()
            #self._dir_artifacts = self.path / 'artifacts'
            #self.dump_params()


    def intercept_std(self):
        return
        #super().intercept_std()
        
        #if self.exp is not None:
            #std_stream = self._stdout_stream
            #if std_stream is not None:
                ##channel_name = 'stdout'
                ##_channel = self.exp._get_channel(channel_name, 'text', ChannelNamespace.SYSTEM)   # is it needed
                ##_channel_writer = ChannelWriter(self.exp, channel_name, ChannelNamespace.SYSTEM)
                ##_channel_writer.write('test\n')
                ##_channel_writer.write('test2\n')
                ##std_stream._writers.append(_channel_writer)
                
                #_channel_writer = StdOutWithUpload(self.exp)
                #_channel_writer.write('test\n')
                #_channel_writer.write('test2\n')
                #std_stream._writers.append(_channel_writer)
                

            #std_stream = self._stderr_stream
            #if self._stderr_stream is not None:
                ##channel_name = 'stderr'
                ##_channel = self.exp._get_channel(channel_name, 'text', ChannelNamespace.SYSTEM)   # is it needed
                ##_channel_writer = ChannelWriter(self.exp, channel_name, ChannelNamespace.SYSTEM)
                ##_channel_writer.write('test3\n')
                ##_channel_writer.write('test4\n')
                ##std_stream._writers.append(_channel_writer)
                
                #std_stream._writers.append(StdErrWithUpload(self.exp))
        
        ##self._stdout_stream = None
        ##self._stderr_stream = None

        ##if not is_notebook():
            ##if self.log_stdout:
                ##fn_log = self.path / 'stdout.txt'
                ##filewriter = FileWriter(fn_log)
                ##self._stdout_stream = StdOutStream([filewriter])
            ##if self.log_stderr:
                ##fn_log = self.path / 'stderr.txt'
                ##filewriter = FileWriter(fn_log)
                ##self._stdout_stream = StdErrStream([filewriter])


    def stop(self):
        print('NeptuneLogger stopping... ')
        print(self.internal)
        if self.internal is not None:
            self.internal.stop()
            print('Stopped.')
        
        #if self._stdout_stream:
            #self._stdout_stream.close()
        #if self._stderr_stream:
            #self._stderr_stream.close()

    def log_artifact(self, artifact_filename, destination=None, local_only=False):
        if not local_only and self.internal is not None:
            self.internal.log_artifact(artifact_filename, destination)

    def save_and_log_artifact(self, value, filename='example.txt', local_only=False):
        """
        Save string as file and send it to remote.
        """
        if not local_only and self.internal is not None:
            artifact_filename = self._dir_artifacts / filename
            self.internal.log_artifact(artifact_filename)

    def set_property(self, key, value):
        if self.internal is not None:
            self.internal.set_property(key, value)

    def append_tag(self, tag, *tags):
        if self.internal is not None:
            self.internal.append_tag(tag, *tags)


    def log_metric(self, name, x, y=None):
        if self.internal is not None:
            self.internal.send_metric(name, x, y)


class StdStreamWithUpload(object):

    def __init__(self, experiment, channel_name):
        # pylint:disable=protected-access
        self._channel = experiment._get_channel(channel_name, 'text', ChannelNamespace.SYSTEM)
        self._channel_writer = ChannelWriter(experiment, channel_name, ChannelNamespace.SYSTEM)

    def write(self, data):
        try:
            self._channel_writer.write(data)
        # pylint:disable=bare-except
        except:
            pass

class StdOutWithUpload(StdStreamWithUpload):

    def __init__(self, experiment):
        super(StdOutWithUpload, self).__init__(experiment, 'stdout')


class StdErrWithUpload(StdStreamWithUpload):

    def __init__(self, experiment):
        super(StdErrWithUpload, self).__init__(experiment, 'stderr')

