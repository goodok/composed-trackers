import os
import warnings
import tempfile
import traceback

from .utils.registry import Registry, build_from_cfg
from .trackers.simple import SimpleTracker
from .trackers.neptune import NeptuneTracker
from .trackers.base import BaseTracker

from .utils.log import print_color

TRACKERS = Registry('Trackers')
TRACKERS.register_module(SimpleTracker)
TRACKERS.register_module(NeptuneTracker)


class ComposedTrackers(BaseTracker):
    def __init__(self, name='name', description='Composed trackers', tags=[], params={}, debug=False, initialize_fn=None, **cfg):
        self.name = name
        self.description = description
        self.tags = tags
        self.params = dict(params)

        self.debug = debug

        self.cfg = cfg

        self._initialize_fn = initialize_fn

        self.initialize()

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
        self.trackers = []

        default_keys = ['name', 'description', 'tags', 'params', 'debug']
        default_args = dict([(key, getattr(self, key)) for key in default_keys])
        suggested_id = None

        for tracker_name in self.cfg['trackers']:
            if isinstance(tracker_name, BaseTracker):
                tracker = tracker_name
            else:
                cfg = default_args.copy()
                cfg['type'] = tracker_name
                if suggested_id is not None:
                    cfg['exp_id'] = suggested_id
                cfg.update(self.cfg[tracker_name])

                tracker = build_from_cfg(cfg, TRACKERS)

            if suggested_id is None:
                suggested_id = tracker.exp_id

            self.trackers.append(tracker)

    def describe(self, ids_only=False):
        if not ids_only:
            print('ComposedTrackers description:')
            keys = ['name', 'description', 'tags', 'debug']
            for key in keys:
                print(f'  {key:12}:', getattr(self, key, None))

            for tracker in self.trackers:
                print()
                tracker.describe()
        else:
            for tracker in self.trackers:
                print(f'{tracker.__class__.__name__:15}:', end=' ')
                print_color(f'{tracker.exp_id}', 'green')
        print()

    def stop(self):
        for tracker in self.trackers:
            try:
                tracker.stop()
            except Exception as e:
                warnings.warn(f"Can't .stop for tracker {tracker}. {e}", UserWarning)

    def set_property(self, key, value):
        for tracker in self.trackers:
            try:
                tracker.set_property(key, value)
            except Exception as e:
                warnings.warn(f"Can't .set_property for tracker {tracker}. {e}", UserWarning)

    def append_tag(self, tag, *tags):
        for tracker in self.trackers:
            try:
                tracker.append_tag(tag, *tags)
            except Exception as e:
                warnings.warn(f"Can't .append_tag for tracker {tracker}. {e}", UserWarning)

    def log_metric(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        if index is None:
            assert autoincrement_index is True, 'Only autoincrement of index is possible for some loggers.'

        for tracker in self.trackers:
            try:
                tracker.log_metric(name, value, index, timestamp, autoincrement_index)
            except Exception as e:
                warnings.warn(f"Can't .log_metric for tracker {tracker}. {e}", UserWarning)
                print(e)
                traceback.print_exc()

    def log_text(self, name, value, index=None, timestamp=None, autoincrement_index=True):
        if index is None:
            assert autoincrement_index is True, 'Only autoincrement of index is possible for some loggers.'
        for tracker in self.trackers:
            try:
                tracker.log_text(name, value, index, timestamp, autoincrement_index)
            except Exception as e:
                warnings.warn(f"Can't .log_text for tracker {tracker}. {e}", UserWarning)
                print(e)
                traceback.print_exc()

    def log_artifact(self, filename, destination=None):
        for tracker in self.trackers:
            try:
                tracker.log_artifact(filename, destination)
            except Exception as e:
                warnings.warn(f"Can't .log_artifact for tracker {tracker} {e}", UserWarning)

    def log_text_as_artifact(self, text, destination=None, existed_temp_file=None):
        fd = None
        try:
            fd, path = tempfile.mkstemp()

            with os.fdopen(fd, 'w') as tmp:
                tmp.write(text)

            for tracker in self.trackers:
                try:
                    tracker.log_artifact(path, destination)
                except Exception as e:
                    warnings.warn(f"Can't .log_artifact for tracker {tracker}. {e}", UserWarning)
        except Exception as e:
            warnings.warn(f"Can't .log_text_as_artifact for composer {self} {e}", UserWarning)
        finally:
            if fd is not None:
                os.remove(path)

    @property
    def path(self):
        for tracker in self.trackers:
            path = getattr(tracker, 'path', None)
            if path is not None:
                return path

    # aliases
    save_and_log_artifact = log_text_as_artifact


TRACKERS.register_module(ComposedTrackers)
