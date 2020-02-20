import warnings

# TODO:
#    - save initial self.properties
#    - log_text
#    - set_property


try:
    import mlflow
except ImportError:
    raise ImportError('Missing mlflow package.')

from .base import BaseTracker


class MLFlowTracker(BaseTracker):

    def __init__(self, name='name', description='Trackers', tags=[],
                 params={},
                 properties={},
                 exp_id=None,
                 tracking_uri=None,
                 registry_uri=None,
                 **kwargs):

        self.name = name
        self.description = description
        self.tags = tags

        self.params = params
        self.properties = properties

        self.exp_id = exp_id

        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

        self._kwargs = kwargs

        self.initialized = False
        self.initialize()

    def describe(self):
        print(self.__class__.__name__)
        print('    exp_id:', self.exp_id)
        print('    _expt_id:', self._expt_id)
        print('    _run_id:', self._run_id)
        print('   tracking_uri:', self.tracking_uri)

    def initialize(self):

        self.internal_handler = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)

        expt = self.internal_handler.get_experiment_by_name(self.name)
        if expt:
            self._expt_id = expt.experiment_id
        else:
            warnings.warn(f"Experiment with name {self.name} not found. Creating it.")
            self._expt_id = self.internal_handler.create_experiment(name=self.name)

        tags = dict([(k, True) for k in self.tags])
        run = self.internal_handler.create_run(experiment_id=self._expt_id, tags=tags)

        self._run_id = run.info._run_id

        self.initialized = True

        self._log_params(self.params)

        # TODO: save initial self.properties

    def _log_params(self, params):
        for k, v in params.items():
            self.internal_handler.log_param(self._run_id, k, v)

    def stop(self):
        print('MLFlowTracker stopping... ', end=' ')
        status = 'FINISHED'
        self.internal_handler.set_terminated(self._run_id, status)
        print('Ok.')

    def log_artifact(self, artifact_filename, destination=None, local_only=False):
        if not local_only:
            self.internal_handler.log_artifact(self._run_id, artifact_filename, destination)

    def save_and_log_artifact(self, value, filename='example.txt', local_only=False):
        """
        Save string as file and send it to remote.
        """
        if not local_only and self.internal_handler is not None:
            artifact_filename = self._dir_artifacts / filename
            self.internal_handler.log_artifact(artifact_filename)

    def set_property(self, key, value):
        # TODO: emulate by artifacts ??
        pass

    def append_tag(self, tag, *tags):
        if isinstance(tag, list):
            tags_list = tag
        else:
            tags_list = [tag] + list(tags)

        tags_set = set(tags_list)
        for k in tags_set:
            self.internal_handler.set_tag(self._run_id, k, True)

    def log_metric(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        self.internal_handler.log_metric(self._run_id, name, value, index)

    def log_text(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        # TODO: emulate by artifacts ??
        pass
