import script_Rocha_2020
import time
import pandas as pd
import matplotlib.pyplot as plt
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

script_Rocha_2020.main()

end = time.time()


print("\nTempo total de simulação:" + str(end - start) + " [s]")