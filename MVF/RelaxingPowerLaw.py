import script_Pinto
import time
import pandas as pd
import matplotlib.pyplot as plt
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

print("\nIniciando simulação")

data = script_Pinto.RK4Solver(
    physicalParameters = script_Pinto.PhysicalParameters(
        height = 0.21, # (m)
        initial_conc = 0.1391,
        particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
        particle_esphericity = 0.8,
        solid_density = 2709, # (kg/m3)
        fluid_density = 891.4, # (kg/m3)
        max_conc = 0.5,
        powerLawFluid_M = 30.13,
        powerLawFluid_n = 0.21,
        yield_tension = 20.88
    ),
    numericalParameters = script_Pinto.NumericalParameters(
        z_divs = 220,
        total_time = 31536000, #365 dias #4320000,
        timestep = 100,
        maxResidual = 0.000000001
    ),
    constantParameters = script_Pinto.ConstantParameters(
        B = 0.02055,
        beta = 3.52, # Pressao nos solidos
        ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
        p_ref = 0.26, # Pressao nos solidos
        relax_time = 48620
    )
)

end = time.time()

pd.DataFrame(data).to_csv("MVF/temporaryFiles/resultadosPreliminares.csv")

print("\nTempo total de simulação:" + str(end - start) + " [s]")