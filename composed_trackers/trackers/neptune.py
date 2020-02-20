import os
# import warnings

try:
    import neptune
    from neptune.internal.streams.channel_writer import ChannelWriter
    from neptune.internal.channels.channels import ChannelNamespace
except ImportError:
    raise ImportError('Missing mlflow package.')


from .base import BaseTracker


class NeptuneTracker(BaseTracker):

    def __init__(self, name='name', description='Trackers', tags=[], offline=False,
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
        self.offline = offline
        self.params = params
        self.properties = properties

        self.project = project
        self.fn_token = fn_token

        self.exp_id = exp_id

        kwargs['upload_stdout'] = upload_stdout
        kwargs['upload_stderr'] = upload_stderr

        self._kwargs = kwargs

        self.initialized = False
        self.initialize()

    def describe(self):
        print(self.__class__.__name__)
        print('    exp_id:', self.exp_id)
        print('   project:', self.project)

    def initialize(self):

        if self.fn_token is not None:
            with open(os.path.expanduser(self.fn_token), 'r') as f:
                token = f.readline().splitlines()[0]
                os.environ['NEPTUNE_API_TOKEN'] = token

        if self.offline:
            neptune.init(project_qualified_name='dry-run/project',
                         backend=neptune.OfflineBackend())
        else:
            neptune.init(project_qualified_name=self.project)

        self.internal_handler = neptune.create_experiment(name=self.name,
                                                          params=self.params,
                                                          properties=self.properties,
                                                          tags=tuple(self.tags),
                                                          description=self.description,
                                                          # upload_source_files=self.upload_source_files,
                                                          **self._kwargs)
        exp_id = self.internal_handler.id
        if isinstance(exp_id, str):
            self.exp_id = exp_id
        self.initialized = True

    def intercept_std(self):
        return

    def stop(self):
        print('NeptuneTracker stopping... ', end=' ')
        self.internal_handler.stop()
        print('Ok.')

        # if self._stdout_stream:
        #    self._stdout_stream.close()
        # if self._stderr_stream:
        #    self._stderr_stream.close()

    def log_artifact(self, artifact_filename, destination=None, local_only=False):
        if not local_only:
            self.internal_handler.log_artifact(artifact_filename, destination)

    def save_and_log_artifact(self, value, filename='example.txt', local_only=False):
        """
        Save string as file and send it to remote.
        """
        if not local_only and self.internal_handler is not None:
            artifact_filename = self._dir_artifacts / filename
            self.internal_handler.log_artifact(artifact_filename)

    def set_property(self, key, value):
        self.internal_handler.set_property(key, value)

    def append_tag(self, tag, *tags):
        self.internal_handler.append_tag(tag, *tags)

    def log_metric(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        # trasnform params from canonic
        if index is None:
            x = value
            y = None
        else:
            x = index
            y = value
        self.internal_handler.send_metric(name, x, y, timestamp)

    def log_text(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        if index is None:
            x = value
            y = None
        else:
            x = index
            y = value
        self.internal_handler.log_text(name, x, y)


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
