import logging
import functools
import pickle
import time
import inspect
from types import FunctionType
from abc import ABCMeta
from threading import Thread

from psutil import virtual_memory, cpu_percent

from ml_recsys_tools.utils import logging_config

logging_config.config()

logger = logging.getLogger(__name__)


class Config:
    """
    Set "use_instrumentation" in some common context
    in order to disable decoration logging.

    I usually leave this on all the time as the profiling is quite
    light weight when not used on very frequently running
    functions (functionality to prevent this is provided below),
    and this also serves as a nice progress indicator to
    know what long-running code is doing at any time.
    """
    use_instrumentation = True
    min_time_seconds = 1
    level = logging.INFO


def variable_info(result):
    """
    Create a string that gives some useful information about an output of a function.
    :param result: anything
    :return: string with some useful information about the result.
    """
    if hasattr(result, 'shape'):
        shape_str = 'shape: %s' % str(result.shape)
    elif isinstance(result, tuple) and len(result) <= 3:
        shape_str = 'tuple: (' + ','.join([variable_info(el) for el in result]) + ')'
    elif hasattr(result, '__len__'):
        shape_str = 'len: %s' % str(len(result))
    elif hasattr(result, '__dict__'):
        shape_str = 'keys: %s' % str(len(result.__dict__.keys()))
    else:
        result_str = str(result)
        shape_str = result_str if len(result_str) <= 50 else (str(result)[:50] + '...')

    return str(type(result)) + ', ' + shape_str


def log_time_and_shape(fn):
    """
    Decorates the function to log it's run time, resource usage (CPU and memory),
    and output shape / info.
    The logging statements are indented according to the stack depth in order to have a
    visual representation of depth.
    :param fn: function to be decorated.
    :return: decorated function.
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        with ResourceMonitor() as monitor:
            result = fn(*args, **kwargs)

        if monitor.elapsed >= Config.min_time_seconds:
            msg = (' ' * get_stack_depth() +
                   f'{function_name_with_class(fn)}, elapsed: {monitor.elapsed:.2f} sec, '
                   f'returned: {variable_info(result)}, '
                   f'sys mem: {monitor.current_memory}%'
                   f'(peak:{monitor.peak_memory}%) '
                   f'cpu:{int(monitor.avg_cpu_load)}%'
                   )

            logger.log(Config.level, msg)

        return result

    return inner if Config.use_instrumentation else fn



class ResourceMonitor:
    """
    Class for monitoring the CPU and memory of a function call.
    """

    def __init__(self, interval=0.2):
        self.interval = interval
        self._init_counters()
        self._thread = None
        self._run_condition = False
        self.elapsed = None

    def _init_counters(self):
        self._start_time = time.time()
        self.current_memory = 0
        self.peak_memory = 0
        self.avg_cpu_load = 0
        self._n_measurements = 0

    def __del__(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    @staticmethod
    def _current():
        try:
            return virtual_memory().percent, cpu_percent()
        except KeyError:
            # for some reason there's a KeyError: ('psutil',) in psutil
            return 0, 0
        except Exception as e:
            logger.exception(e)
            return 0, 0

    def _measure(self):
        cur_mem, cur_cpu = self._current()
        self.current_memory = cur_mem
        self.peak_memory = max(self.peak_memory, cur_mem)
        self.avg_cpu_load = (self.avg_cpu_load * self._n_measurements + cur_cpu) / \
                            (self._n_measurements + 1)
        self._n_measurements += 1

    def _thread_loop(self):
        while self._run_condition:
            self._measure()
            time.sleep(self.interval)

    def start(self):
        self._init_counters()
        if self._thread is not None:
            self._thread.join(0)
        self._run_condition = True
        self._thread = Thread(target=self._thread_loop, name='ResourceMonitor')
        self._thread.start()
        return self

    def stop(self):
        self.elapsed = time.time() - self._start_time
        self._run_condition = False
        self._measure()


def get_stack_depth():
    try:
        return len(inspect.stack(context=0))
    except (IndexError, RuntimeError) as e:
        # there is a bug in inspect module: https://github.com/ipython/ipython/issues/1456/
        # another one: https://bugs.python.org/issue13487
        return 0


def function_name_with_class(fn):
    cls = get_class_that_defined_method(fn)
    cls_str = cls.__name__ + '.' if cls else ''
    return cls_str + fn.__name__


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
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0], None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def decorate_all_metaclass(decorator):
    """
    Decorate all methods and classmethods (unless excluded) with the same decorator.

    altered from https://stackoverflow.com/questions/10067262/automatically-decorating-every-instance-method-in-a-class
    """

    # check if an object should be decorated
    def do_decorate(attr, value):
        return ('__' not in attr and
                isinstance(value, (FunctionType, classmethod)) and
                getattr(value, 'decorate', True))

    class DecorateAll(ABCMeta):
        def __new__(cls, name, bases, dct):
            if dct.get('decorate', True):
                for attr, value in dct.items():
                    if do_decorate(attr, value):
                        if isinstance(value, classmethod):
                            dct[attr] = classmethod(decorator(value.__func__))
                        else:
                            dct[attr] = decorator(value)
            return super(DecorateAll, cls).__new__(cls, name, bases, dct)

        def __setattr__(self, attr, value):
            if do_decorate(attr, value):
                value = decorator(value)
            super(DecorateAll, self).__setattr__(attr, value)

    return DecorateAll


class LogLongCallsMeta(metaclass=decorate_all_metaclass(log_time_and_shape)):
    """
        Class to provide automatic logging decoration of all methods for extending classes.

        Note on overhead for frequently invoked methods:
            If your class includes methods that are invoked very frequently,
            e.g. thousands / millions of times per second, you should use `do_not_decorate`
            decorator in order to prevent the logging decorator being applied to those methods,
            since the overhead of monitoring will be substantial in those cases.
        """

    @property
    def logging_decorator(self):
        """
        this is for decorating inner scope functions
        :return: the logging decorator if verbose is True, empty decorator otherwise
        """
        return log_time_and_shape

    @staticmethod
    def do_not_decorate(f):
        """
        this is for excluding functions from decorating
        :return: the logging decorator if verbose is True, empty decorator otherwise
        """
        setattr(f, 'decorate', False)
        return f


def log_errors(message=None, return_on_error=None):
    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            nonlocal message, return_on_error
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.exception(e)
                fn_str = function_name_with_class(fn)
                msg_str = ', Message: %s' % message if message else ''
                logger.error('Failed: %s, Error: %s %s' %(fn_str, str(e), msg_str))
                return return_on_error
        return inner
    return decorator


def pickle_size_mb(obj):
    return pickle.dumps(obj).__sizeof__() / 1e6