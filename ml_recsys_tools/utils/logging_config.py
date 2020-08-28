import logging

import sys

FORMAT = logging.Formatter(
    '[%(asctime)s:%(levelname)s:%(module)s] %(message)s', datefmt='%H:%M:%S')
LEVEL = logging.INFO


def console_handler(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(FORMAT)
    return handler


def config():
    logging.basicConfig(level=LEVEL, handlers=[console_handler()])
