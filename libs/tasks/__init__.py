# Expose the broker symbol so `taskiq worker tasks:broker` works
from common.broker import broker  # noqa: F401

# Import the tasks module so TaskIQ discovers @broker.task on package import
from . import tasks as _tasks  # noqa: F401
