"""Process-local admission and dependency circuit breaker; no unbounded waiting queue."""
import time
from contextlib import contextmanager
from threading import Lock


class CapacityExceeded(Exception):
    pass


class CircuitOpen(Exception):
    pass


class Admission:
    def __init__(self, capacity):
        self.capacity, self.active = capacity, 0
        self.lock = Lock()

    @contextmanager
    def enter(self):
        with self.lock:
            if self.active >= self.capacity:
                raise CapacityExceeded()
            self.active += 1
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1


class CircuitBreaker:
    def __init__(self, threshold=5, cooldown=30, clock=time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures, self.until, self.probing = 0, 0.0, False
        self.lock = Lock()

    def before(self):
        with self.lock:
            if self.until:
                if self.clock() < self.until or self.probing:
                    raise CircuitOpen()
                self.probing = True  # exactly one half-open probe

    def success(self):
        with self.lock:
            self.failures, self.until, self.probing = 0, 0.0, False

    def failure(self):
        with self.lock:
            self.failures += 1
            if self.probing or self.failures >= self.threshold:
                self.until = self.clock() + self.cooldown
            self.probing = False

    def cancel(self):
        with self.lock:
            self.probing = False
