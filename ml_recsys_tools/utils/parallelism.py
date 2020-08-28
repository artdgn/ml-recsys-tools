import logging
from itertools import islice
import multiprocessing
import multiprocessing.pool

N_CPUS = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)


def batch_generator(iterable, n=1):
    if hasattr(iterable, '__len__'):
        # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    elif hasattr(iterable, '__next__'):
        # https://stackoverflow.com/questions/1915170/split-a-generator-iterable-every-n-items-in-python-splitevery
        i = iter(iterable)
        piece = list(islice(i, n))
        while piece:
            yield piece
            piece = list(islice(i, n))
    else:
        raise ValueError('Iterable is not iterable?')


def map_batches_multiproc(func, iterable, chunksize, multiproc_mode='threads',
                          n_threads=None, threads_per_cpu=1.0):
    if n_threads is None:
        n_threads = int(threads_per_cpu * N_CPUS)

    if hasattr(iterable, '__len__') and len(iterable) <= chunksize:
        return [func(iterable)]

    with pool_type(multiproc_mode)(n_threads) as pool:
        batches = batch_generator(iterable, n=chunksize)
        return list(pool.imap(func, batches))


def pool_type(parallelism_type):
    if 'process' in parallelism_type.lower():
        return multiprocessing.pool.Pool
    elif 'thread' in parallelism_type.lower():
        return multiprocessing.pool.ThreadPool
    else:
        raise ValueError('Unsupported value for "parallelism_type"')
