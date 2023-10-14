import numpy as np
from numpy.linalg import LinAlgError


class MockClass:
    """Very simple mock class for testing purposes."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sum(self):
        raise LinAlgError()


if __name__ == "__main__":
    print(f"Hello World!{np.abs(2)}")
