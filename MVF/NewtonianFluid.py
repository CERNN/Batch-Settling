import script_GL_92
import time
import pandas as pd
import matplotlib.pyplot as plt
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

print("\nIniciando simulação")

data = script_GL_92.CrankNewtSolver()

end = time.time()

pd.DataFrame(data).to_csv("MVF/temporaryFiles/resultadosPreliminares.csv")

print("\nTempo total de simulação:" + str(end - start) + " [s]")