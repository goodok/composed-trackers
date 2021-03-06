
# https://stackoverflow.com/questions/14571090/ipython-redirecting-output-of-a-python-script-to-a-file-like-bash
# https://stackoverflow.com/questions/34145950/is-there-a-way-to-redirect-stderr-to-file-in-jupyter

#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
from pathlib import Path

# from neptune.internal.channels.channels import ChannelNamespace
# from neptune.internal.streams.channel_writer import ChannelWriter


class FileWriter(object):
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(str(self.path), 'w')

    def write(self, v):
        self.f.write(v)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


class StdStream(object):
    def __init__(self, __std_stream__, writers):
        self._stream = __std_stream__
        self._writers = writers

    def write(self, data):
        self._stream.write(data)
        for w in self._writers:
            try:
                w.write(data)
            # pylint:disable=bare-except
            except:
                pass

    def isatty(self):
        # Check if the file is connected to a terminal device:
        return hasattr(self._stream, 'isatty') and self._stream.isatty()

    def flush(self):
        self._stream.flush()
        for w in self._writers:
            try:
                w.flush()
            # pylint:disable=bare-except
            except:
                pass

    def fileno(self):
        "returns the file descriptor of the stream, as a number."
        for w in self._writers:
            try:
                return w.fileno()
            # pylint:disable=bare-except
            except:
                pass

    def close(self):
        self.flush()
        for w in self._writers:
            try:
                return w.close()
            # pylint:disable=bare-except
            except:
                pass


class StdOutStream(StdStream):
    """
    Example
    -------
        if not is_notebook():
            fn_log = logger.path / 'stdout.txt'
            filewriter = FileWriter(fn_log)

            stdout_writer = StdOutStream([filewriter])
        # some code
        # ....

        if not is_notebook():
            stdout_writer.close()
    """
    def __init__(self, writers):
        self._old_stream = sys.stdout
        super().__init__(sys.__stdout__, writers)
        # super().__init__(sys.stdout, writers)
        sys.stdout = self

    def close(self):
        super().close()
        sys.stdout = self._old_stream


class StdErrStream(StdStream):
    def __init__(self, writers):
        self._old_stream = sys.stderr
        super().__init__(sys.__stderr__, writers)
        sys.stderr = self

    def close(self):
        super().close()
        sys.stderr = self._old_stream
