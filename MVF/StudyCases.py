from calendar import c
from cmath import exp
import script_Rocha_2020
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import numpy as np
#python -m timeit -s 'from primes_python import primes' 'primes(1000)'

start = time.time()

# rocha_data = pd.ExcelFile("MVF/RochaData.xlsx")
# dfs = {sheet_name: rocha_data.parse(sheet_name) 
#           for sheet_name in rocha_data.sheet_names}
# num_data = dfs["Numerico"].values
# exp_data = dfs["Experimental"].values

print("\nIniciando simulação")
print(np.array([]).size)
Setups = [
    [
        script_Rocha_2020.PhysicalParameters(
            height = 0.21, # (m)
            # initial_conc = 0.1391,
            initial_conc = 0.07,
            particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
            # particle_diam = 0.00001,
            particle_esphericity = 0.8,
            solid_density = 2709, # (kg/m3)
            fluid_density = 891.4, # (kg/m3)
            max_conc = 0.2, #0.19
            powerLawFluid_M = 30.13,
            powerLawFluid_n = 0.21
        ),
        script_Rocha_2020.NumericalParameters(
            z_divs = 220,
            total_time = 31536000 * 3, #365 dias #4320000,
            timestep = 300,
            maxResidual = 0.000000001,
            indexesToPlot = [5,21,31] #220 dvs
            # indexesToPlot = [0,7,28,42] #100 dvs
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        np.array([]),
        np.array([]),
    ],
    [
        script_Rocha_2020.PhysicalParameters(
            height = 0.21, # (m)
            initial_conc = 0.1391,
            # initial_conc = 0.07,
            particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
            particle_esphericity = 0.8,
            solid_density = 2709, # (kg/m3)
            fluid_density = 891.4, # (kg/m3)
            max_conc = 0.2, #0.19
            powerLawFluid_M = 30.13,
            powerLawFluid_n = 0.21
            # powerLawFluid_n = 0.3
        ),
        script_Rocha_2020.NumericalParameters(
            z_divs = 220,
            total_time = 31536000 * 3, #365 dias #4320000,
            timestep = 300,
            maxResidual = 0.000000001,
            indexesToPlot = [5,21,31] #220 dvs
            # indexesToPlot = [0,7,28,42] #100 dvs
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        np.array([]),
        np.array([]),
    ],
    [
        script_Rocha_2020.PhysicalParameters(
            height = 0.21, # (m)
            initial_conc = 0.17,
            # initial_conc = 0.1391,
            particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
            # particle_diam = 0.0001,
            particle_esphericity = 0.8,
            solid_density = 2709, # (kg/m3)
            fluid_density = 1200, # (kg/m3)
            max_conc = 0.2, #0.19
            powerLawFluid_M = 30.13,
            powerLawFluid_n = 0.21
            # powerLawFluid_n = 0.4
        ),
        script_Rocha_2020.NumericalParameters(
            z_divs = 220,
            total_time = 31536000 * 3, #365 dias #4320000,
            timestep = 300,
            maxResidual = 0.000000001,
            indexesToPlot = [5,21,31] #220 dvs
            # indexesToPlot = [0,7,28,42] #100 dvs
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        script_Rocha_2020.ConstantParameters(
            delta = 0.58, # Permeabilidade - Rocha (2020)
            k0 = 46, # Permeabilidade - Rocha (2020)
            beta = 0.19, # Pressao nos solidos
            ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
            p_ref = 18.62 # Pressao nos solidos
        ),
        np.array([]),
        np.array([]),
    ],
    # [
    #     script_Rocha_2020.PhysicalParameters(
    #         height = 0.21, # (m)
    #         initial_conc = 0.11,
    #         particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    #         # particle_diam = 0.0001,
    #         particle_esphericity = 0.8,
    #         solid_density = 2709, # (kg/m3)
    #         fluid_density = 891.4, # (kg/m3)
    #         max_conc = 0.2, #0.19
    #         powerLawFluid_M = 30.13,
    #         powerLawFluid_n = 0.21
    #         # powerLawFluid_n = 0.35
    #     ),
    #     script_Rocha_2020.NumericalParameters(
    #         z_divs = 220,
    #         total_time = 31536000 * 3, #365 dias #4320000,
    #         timestep = 300,
    #         maxResidual = 0.000000001,
    #         indexesToPlot = [5,21,31] #220 dvs
    #         # indexesToPlot = [0,7,28,42] #100 dvs
    #     ),
    #     script_Rocha_2020.ConstantParameters(
    #         delta = 0.58, # Permeabilidade - Rocha (2020)
    #         k0 = 46, # Permeabilidade - Rocha (2020)
    #         beta = 0.19, # Pressao nos solidos
    #         ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #         p_ref = 18.62 # Pressao nos solidos
    #     ),
    #     script_Rocha_2020.ConstantParameters(
    #         delta = 0.58, # Permeabilidade - Rocha (2020)
    #         k0 = 46, # Permeabilidade - Rocha (2020)
    #         beta = 0.19, # Pressao nos solidos
    #         ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #         p_ref = 18.62 # Pressao nos solidos
    #     ),
    #     np.array([]),
    #     np.array([]),
    # ],
    #     [
    #     script_Rocha_2020.PhysicalParameters(
    #         height = 0.21, # (m)
    #         initial_conc = 0.17,
    #         # initial_conc = 0.1391,
    #         particle_diam = 0.0000408, # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    #         # particle_diam = 0.00015,
    #         particle_esphericity = 0.8,
    #         solid_density = 2709, # (kg/m3)
    #         fluid_density = 1200, # (kg/m3)
    #         max_conc = 0.2, #0.19
    #         powerLawFluid_M = 30.13,
    #         powerLawFluid_n = 0.21
    #         # powerLawFluid_n = 0.4
    #     ),
    #     script_Rocha_2020.NumericalParameters(
    #         z_divs = 220,
    #         total_time = 31536000, #365 dias #4320000,
    #         timestep = 300,
    #         maxResidual = 0.000000001,
    #         indexesToPlot = [5,21,31] #220 dvs
    #         # indexesToPlot = [0,7,28,42] #100 dvs
    #     ),
    #     script_Rocha_2020.ConstantParameters(
    #         delta = 0.58, # Permeabilidade - Rocha (2020)
    #         k0 = 46, # Permeabilidade - Rocha (2020)
    #         beta = 0.19, # Pressao nos solidos
    #         ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #         p_ref = 18.62 # Pressao nos solidos
    #     ),
    #     script_Rocha_2020.ConstantParameters(
    #         delta = 0.58, # Permeabilidade - Rocha (2020)
    #         k0 = 46, # Permeabilidade - Rocha (2020)
    #         beta = 0.19, # Pressao nos solidos
    #         ref_conc = 0.145, #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    #         p_ref = 18.62 # Pressao nos solidos
    #     ),
    #     np.array([]),
    #     np.array([]),
    # ],
]

# linestyles = ['','dashed','dotted','dashdot','solid']
# markerstyles = ['_','','','','']
linestyles = ['solid','dashed','dashdot','dotted']
markerstyles = ['','','','']

DATA = []

for setup in Setups:
    data = script_Rocha_2020.RK4Solver(
        physicalParameters = setup[0],
        numericalParameters = setup[1],
        packingParameters = setup[2],
        clarifiedParameters = setup[3],
        Rocha_exp_data = setup[4],
        Rocha_num_data = setup[5],
    )
    DATA.append(data)

labels = ['Fluido 1','Fluido 2', 'Fluido 3','Fluido 4','Fluido 5']
# labels = ['Particulado 1','Particulado 2', 'Particulado 3','Particulado 4','Particulado 5']
# labels = ['Suspensão 1','Suspensão 2', 'Suspensão 3','Suspensão 4','Suspensão 5']
lCount = 0

for data in DATA:
    indexToPlot = [5,21,31]
    Time = np.arange(len(data))
    colors = ['gray','blue','magenta','red','cyan','green']
    counter = 0
    # if lCount == 0:
    #     Position = 0.21 * 0.5 / 220 + np.arange(220) * 0.21 / 220
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     Position, Time = np.meshgrid(Position, Time)
    #     surf = ax.plot_surface(Position, Time, data)
    #     plt.show()
    for index in indexToPlot:
        PlotData = []
        for concentrationData in data:
            PlotData.append(concentrationData[index])
        positionToPlot = 0.21 * (1 + 2 * index) / (2 * 220)
    
        plt.plot(Time,PlotData, color=colors[counter], label= "n=" + str(index) + ", z=" + str("{:.2f}".format(positionToPlot * 100)) + " cm" + " - " + labels[lCount], linestyle = linestyles[lCount], marker=markerstyles[lCount])
        counter += 1
        
    lCount += 1

plt.xlabel('Tempo [Dias]')
plt.xlim(0.400 * 3)
plt.xscale('log')
plt.ylabel('Concentração')
plt.ylim(0.06,0.22)
# plt.title('Conc_max = ' + str(0.2))
plt.grid()
plt.legend(bbox_to_anchor = (1, 1) )

plt.savefig('MVF/temporaryFiles/Concentration.png')

plt.show()
    

end = time.time()

pd.DataFrame(data).to_csv("MVF/temporaryFiles/resultadosPreliminares.csv")

print("\nTempo total de simulação:" + str(end - start) + " [s]")
