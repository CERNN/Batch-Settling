import script_Rocha_2020
import time
import pandas as pd
import matplotlib.pyplot as plt
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

print("\nIniciando simulação")

""" data = script_Rocha_2020.EulerSolver(
    physicalParameters = script_Rocha_2020.PhysicalParameters(
        height = 0.21, # (m)
        initial_conc = 0.1391,
        particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
        solid_density = 2709, # (kg/m3)
        fluid_density = 1145, # (kg/m3)
        max_conc = 0.2 ,
        powerLawFluid_M = 30.13,
        powerLawFluid_n = 0.21
    ),
    numericalParameters = script_Rocha_2020.NumericalParameters(
        z_divs = 21,
        total_time = 432000,
        timestep = 0.1
    ),
    constantParameters = script_Rocha_2020.ConstantParameters(
        delta = 0.58, # Permeabilidade - Rocha (2020)
        k0 = 27.99, # Permeabilidade - Rocha (2020)
        beta = 0.19, # Pressao nos solidos
        ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
        p_ref = 18.62 # Pressao nos solidos

    )
) """

data = script_Rocha_2020.CrankSolver(
    physicalParameters = script_Rocha_2020.PhysicalParameters(
        height = 0.21, # (m)
        initial_conc = 0.1391,
        particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
        solid_density = 2709, # (kg/m3)
        fluid_density = 1145, # (kg/m3)
        max_conc = 0.2 ,
        powerLawFluid_M = 30.13,
        powerLawFluid_n = 0.21
    ),
    numericalParameters = script_Rocha_2020.NumericalParameters(
        z_divs = 220,
        total_time = 31536000, #365 dias #4320000,
        timestep = 1,
        maxResidual = 0.000000001
    ),
    constantParameters = script_Rocha_2020.ConstantParameters(
        delta = 0.58, # Permeabilidade - Rocha (2020)
        k0 = 27.99, # Permeabilidade - Rocha (2020)
        beta = 0.19, # Pressao nos solidos
        ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
        p_ref = 18.62 # Pressao nos solidos
    )
)

end = time.time()

pd.DataFrame(data).to_csv("MVF/temporaryFiles/resultadosPreliminares.csv")

print("\nTempo total de simulação:" + str(end - start) + " [s]")