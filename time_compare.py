import bisect
import random
import timeit
import numpy as np

from quantize import quantize

x = list(range(100000))
n = 1000
random.shuffle(x)
x = np.array(x, dtype=np.float)

def quantize_slow(x, n):
    values = sorted(list(x))
    boundaries = [values[len(x) * p // n] for p in range(n)]

    quantized = [bisect.bisect_left(boundaries, x[i]) for i in range(len(x))]
    return np.array(boundaries), np.array(quantized)


def test():
    boundaries, quantized = quantize(x, n)


def test_slow():
    boundaries, quantized = quantize_slow(x, n)


if __name__ == "__main__":

    print ("len(x) =", len(x), ", n =", n)
    print ("quntize: ", timeit.timeit("test()", setup="from __main__ import test", number=1))
    print ("quntize_slow: ", timeit.timeit("test_slow()", setup="from __main__ import test_slow", number=1))
