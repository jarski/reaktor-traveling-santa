import multiprocessing
import sys
import TravelingSanta as ts

if __name__ == "__main__":
    # The ants rely on fork() sharing the 10,001 x 10,001 matrices copy-on-write,
    # which was the default on macOS in 2018. Python 3.8+ spawns instead.
    multiprocessing.set_start_method("fork")
    ts.solve(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
