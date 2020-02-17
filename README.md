# composed-trackers

Simplified and lightweight interface to experiment tracking systems.

Allows simultaneous and independent local offline tracking and one or more remote ML trackers such as [neptune.ai](https://neptune.ai/), [mlflow.org](https://mlflow.org/docs/latest/tracking.html), [comet.ml](https://www.comet.ml/site/), Tensorboard, [test-tube](https://williamfalcon.github.io/test-tube/) in different combinations. 


## Quick Start

```python
from composed_trackers import ComposedTrackers
from composed_trackers.trackers import NeptuneTracker, SimpleTracker

params_shared = {
    'name': 'Experiment 01 name',
    'description': 'Description',
    'tags': ['examples'],
    'params': {'num_epochs': 10, 'optimizer': 'Adam'},
    'offline': True,    # Switch to "False" if you finish debugging.
}

# Neptune tracker
# https://docs.neptune.ai/
neptune_tracker = NeptuneTracker(project='USER_NAME/PROJECT_NAME', **params_shared)

# Simple offline tracker
simple_tracker = SimpleTracker(exp_id=neptune_tracker.exp_id,    # Try to use neptune_tracker.exp_id
                               root_path='./logs',
                               exp_id_template='EXAM01-{i:03}',
                               **params_shared)

# Create a tracker for simultaneous logging using SimpleTracker and NeptuneTracker.
tracker = ComposedTrackers(
    trackers=[neptune_tracker, simple_tracker],
    **params_shared)


tracker.describe()

# usage

tracker.append_tag('introduction-minimal-example')
n = 117
for i in range(1, n):
    tracker.log_metric('iteration', i)
    tracker.log_metric('loss', 1/i**0.5)
    tracker.log_text('magic values', 'magic value {}'.format(0.95*i**2))
tracker.set_property('n_iterations', n)

tracker.log_text_as_artifact('Hello', 'summary.txt')

# finish experiment
tracker.stop()

tracker.describe(ids_only=True)
```

## Examples

[QuickStart](https://github.com/goodok/composed-trackers/tree/master/examples/minimal)

[Kekas](https://github.com/goodok/composed-trackers/tree/master/examples/kekas)

