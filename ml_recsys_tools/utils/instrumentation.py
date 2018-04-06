import logging
import functools
import time
import inspect
from abc import ABC
from threading import Thread
from psutil import virtual_memory

from ml_recsys_tools.utils.logger import simple_logger

# set this to logging.DEBUG to silence the printouts
WRAPPERS_LOGGING_LEVEL = logging.INFO

# set this to longer time to reduce printouts of short calls
MIN_TIME_TO_LOG = 1


def variable_info(result):
    if hasattr(result, 'shape'):
        shape_str = 'shape: %s' % str(result.shape)
    elif isinstance(result, tuple) and len(result) <= 3:
        shape_str = 'tuple: (' + ','.join([variable_info(el) for el in result]) + ')'
    elif hasattr(result, '__len__'):
        shape_str = 'len: %s' % str(len(result))
    else:
        shape_str = str(result)[:50] + '...'

    ret_str = str(type(result)) + ', ' + shape_str
    return ret_str


def log_time_and_shape(fn):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        mem_monitor = MaxMemoryMonitor().start()

        start = time.time()

        result = fn(*args, **kwargs)

        elapsed = time.time() - start
        duration_str = '%.2f' % elapsed

        cur_mem, peak_mem = mem_monitor.stop()

        mem_str = '%s%%(peak:%s%%)' % (cur_mem, peak_mem)

        ret_str = variable_info(result)

        stack_depth = get_stack_depth()

        fn_str = class_name(fn) + fn.__name__

        msg = ' ' * stack_depth + \
              '%s, elapsed: %s, returned: %s, sys mem: %s' % \
              (fn_str, duration_str, ret_str, mem_str)

        if elapsed >= MIN_TIME_TO_LOG:
            simple_logger.log(WRAPPERS_LOGGING_LEVEL, msg)

        return result

    return inner


def timer_deco(fn):
    log_format = 'function %s: %s s'

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        simple_logger.log(WRAPPERS_LOGGING_LEVEL, log_format,
                          fn.__name__, duration)
        return result

    return inner


class MaxMemoryMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_memory = None
        self.thread = None
        self._run_condition = False

    def __del__(self):
        self._run_condition = False

    @staticmethod
    def _current():
        return virtual_memory().percent

    def _measure_peak(self):
        self.peak_memory = max(self.peak_memory, self._current())

    def _thread_loop(self):
        while self._run_condition:
            self._measure_peak()
            time.sleep(self.interval)

    def start(self):
        self._run_condition = True
        self.peak_memory = 0
        self.thread = Thread(target=self._thread_loop, name='MaxMemoryMonitor')
        self.thread.start()
        return self

    def stop(self):
        self._run_condition = False
        self._measure_peak()
        return self._current(), self.peak_memory


def get_stack_depth():
    try:
        return len(inspect.stack())
    except IndexError as e:
        # there is a bug in inspect module: https://github.com/ipython/ipython/issues/1456/
        return 0


def class_name(fn):
    cls = get_class_that_defined_method(fn)
    cls_str = cls.__name__ + '.' if cls else ''
    return cls_str


def get_class_that_defined_method(meth):
    # from https://stackoverflow.com/questions/3589311/
    # get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
    # modified to return first parent in reverse MRO
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__)[::-1]:
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def collect_named_init_params(cls, skip_empty=True, ignore_classes=(object, ABC)):
    """
    a method to get all named params from all classed in this class' MRO
    can be used to infer the possible search space for hyperparam optimization

    :param skip_empty: whether to skip classes with no named parameters
    :return: a nested dict of {class-name: dict of {named-init-params: default values}}
    """
    params = {}
    for c in inspect.getmro(cls):
        if c not in ignore_classes:
            named_params = [
                p for p in inspect.signature(c.__init__).parameters.values()
                if p.name != 'self'
                   and p.kind != p.VAR_KEYWORD
                   and p.kind != p.VAR_POSITIONAL]

            if skip_empty and not named_params:
                continue

            params[c.__name__] = {
                p.name: p.default
                if p.default is not p.empty else ''
                for p in named_params}
    return params