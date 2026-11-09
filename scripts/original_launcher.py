"""Run one of the authors' original scripts unchanged (Python 2.7, DyNet 2.1.2).

Usage: python original_launcher.py <path/to/trainer.py> [its arguments]

The authors built against their own DyNet fork. Mainline DyNet 2.1.2 differs
in one function the code uses, so two shims are applied here, and nothing in
the original files is edited:
  1. the fork's flip_gradient(x, ro) (identity forward, gradient times -ro
     backward) is mapped to mainline scale_gradient(x, -ro); mainline's own
     flip_gradient(x) takes no scale;
  2. tensorboard_logger, used only to draw TensorBoard curves, is replaced by
     no-op functions.
"""
import os
import runpy
import sys
import types

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]

import dynet as dy  # noqa: E402  (reads the --dynet-* options from sys.argv)

_mainline_flip = dy.flip_gradient


def _flip_gradient(x, scale=None):
    if scale is None:
        return _mainline_flip(x)
    return dy.scale_gradient(x, -float(scale))


dy.flip_gradient = _flip_gradient

tb = types.ModuleType("tensorboard_logger")
tb.configure = lambda *a, **k: None
tb.log_value = lambda *a, **k: None
sys.modules["tensorboard_logger"] = tb

os.chdir(os.path.dirname(script))
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
