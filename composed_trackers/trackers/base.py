# from abc import ABC, abstractmethod
# import warnings

# TODO:
# abstractmethod politics
# exp_id - abstarct

# aliases as methods


class BaseTracker():
    """Base class for trackers."""

    def __init__(self):
        pass

    # @abstractmethod
    def initialize(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def set_property(self, key, value):
        raise NotImplementedError

    def append_tag(self, tag, *tags):
        raise NotImplementedError

    def log_metric(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        raise NotImplementedError

    def log_text(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        raise NotImplementedError

    def log_artifact(self, filename, destination=None):
        raise NotImplementedError

    def log_text_as_artifact(self, text, destination=None, existed_temp_file=None):
        raise NotImplementedError

    def delete_artifacts(self, path):
        raise NotImplementedError

    # aliases
    def send_metric(self, *args, **kwargs):
        self.log_metric(*args, **kwargs)

    def send_artifact(self, *args, **kwargs):
        self.log_artifact(*args, **kwargs)

    def send_text(self, *args, **kwargs):
        self.send_text(*args, **kwargs)
