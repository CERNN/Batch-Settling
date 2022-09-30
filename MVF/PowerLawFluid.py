from cmath import exp
import script_Rocha_2020
import time
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

print("\nIniciando simulação")

rocha_data = pd.ExcelFile("MVF/RochaData.xlsx")
dfs = {sheet_name: rocha_data.parse(sheet_name) 
          for sheet_name in rocha_data.sheet_names}
num_data = dfs["Numerico"].values
exp_data = dfs["Experimental"].values
# print(num_data[:,0])
# plt.plot(exp_data[:,0],exp_data[:,1])
# plt.plot(exp_data[:,2],exp_data[:,3])
# plt.plot(exp_data[:,4],exp_data[:,5])
# plt.show()

data = script_Rocha_2020.RK4Solver(
    physicalParameters = script_Rocha_2020.PhysicalParameters(
        height = 0.21, # (m)
        initial_conc = 0.1391,
        particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
        particle_esphericity = 0.8,
        solid_density = 2709, # (kg/m3)
        fluid_density = 891.4, # (kg/m3)
        max_conc = 0.2, #0.19
        powerLawFluid_M = 30.13,
        powerLawFluid_n = 0.21
    ),
    numericalParameters = script_Rocha_2020.NumericalParameters(
        z_divs = 220,
        total_time = 31536000, #365 dias #4320000,
        timestep = 300,
        maxResidual = 0.000000001,
        indexesToPlot = [5,21,31]#,41,62,83,125] #220 dvs
        # indexesToPlot = [0,7,28,42] #100 dvs
    ),
    packingParameters = script_Rocha_2020.ConstantParameters(
        delta = 0.58, # Permeabilidade - Rocha (2020)
        k0 = 27.99, # Permeabilidade - Rocha (2020)
        beta = 0.19, # Pressao nos solidos
        ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
        p_ref = 18.62 # Pressao nos solidos
    ),
    clarifiedParameters = script_Rocha_2020.ConstantParameters(
        delta = 0.58, # Permeabilidade - Rocha (2020)
        k0 = 27.99, # Permeabilidade - Rocha (2020)
        beta = 0.19, # Pressao nos solidos
        ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
        p_ref = 18.62 # Pressao nos solidos
    ),
    Rocha_exp_data = exp_data,
    Rocha_num_data = num_data,
    # constantParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 1.04, # Permeabilidade - Rocha (2020)
    #     k0 = 52.67, # Permeabilidade - Rocha (2020)
    #     beta = 1.0, # Pressao nos solidos
    #     ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 3.31 # Pressao nos solidos
    # )
    #Clarificado
    # constantParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 0.58, # Permeabilidade - Rocha (2020)
    #     k0 = 27.99, # Permeabilidade - Rocha (2020)
    #     beta = 0.19, # Pressao nos solidos
    #     ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 18.62 # Pressao nos solidos
    # )
    # constantParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 1.04, # Permeabilidade - Rocha (2020)
    #     k0 = 52.67, # Permeabilidade - Rocha (2020)
    #     beta = 1.0, # Pressao nos solidos
    #     ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 3.31 # Pressao nos solidos
    # )
    # packingParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 1.04, # Permeabilidade - Rocha (2020)
    #     k0 = 52.67, # Permeabilidade - Rocha (2020)
    #     beta = 1.0, # Pressao nos solidos
    #     ref_conc = 0.16, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 3.31 # Pressao nos solidos
    # ),
    # clarifiedParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 1.04, # Permeabilidade - Rocha (2020)
    #     k0 = 52.67, # Permeabilidade - Rocha (2020)
    #     beta = 1.0, # Pressao nos solidos
    #     ref_conc = 0.135, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 3.31 # Pressao nos solidos
    # )
    # Empacotamento
    # clarifiedParameters = script_Rocha_2020.ConstantParameters(
    #     delta = 0.58, # Permeabilidade - Rocha (2020)
    #     k0 = 27.99, # Permeabilidade - Rocha (2020)
    #     beta = 0.19, # Pressao nos solidos
    #     ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #     p_ref = 18.62 # Pressao nos solidos
    # )
)

end = time.time()

pd.DataFrame(data).to_csv("MVF/temporaryFiles/resultadosPreliminares.csv")

print("\nTempo total de simulação:" + str(end - start) + " [s]")
