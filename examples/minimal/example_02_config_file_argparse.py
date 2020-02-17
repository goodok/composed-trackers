#!/usr/bin/env python
# coding: utf-8

#  usage:
#  python example_02_config_file_argparse.py configs/example_02.yaml  --help

#  python example_02_config_file_argparse.py configs/example_02.yaml  --tracker.offline=1



from composed_trackers import Config, build_from_cfg, TRACKERS, is_notebook, load_config_with_shell_updates
from pathlib import Path


cfg = load_config_with_shell_updates()

tracker = build_from_cfg(cfg.tracker, TRACKERS, {'params': cfg.to_flatten_dict(sep='.')})

tracker.describe()

tracker.append_tag('introduction-minimal-example')

n = 117
for i in range(1, n):
    tracker.log_metric('iteration', i)
    tracker.log_metric('loss', 1/i**0.5)
    tracker.log_text('magic values', 'magic value {}'.format(0.95*i**2))
tracker.set_property('n_iterations', n)

tracker.log_text_as_artifact('Hello', 'summary.txt')

# ## Stop
tracker.stop()
tracker.describe(ids_only=True)




