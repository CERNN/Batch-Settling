# cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cimport numpy as np

cdef extern from "math.h":
    double exp(double)

cdef class ConstantParameters:
    cdef double B
    cdef double beta
    cdef double ref_conc
    cdef double p_ref
    cdef double relax_time

    def __init__(self, B, beta, ref_conc, p_ref, relax_time):
        self.B = B
        self.beta = beta
        self.ref_conc = ref_conc
        self.p_ref = p_ref
        self.relax_time = relax_time

cdef class PhysicalParameters:
    cdef double L
    cdef double initial_conc
    cdef double particle_diam
    cdef double particle_esphericity
    cdef double solid_density
    cdef double fluid_density
    cdef double max_conc
    cdef double M
    cdef double n
    cdef double yield_tension

    def __init__(self, height,initial_conc,particle_diam,particle_esphericity,solid_density,fluid_density,max_conc,powerLawFluid_M,powerLawFluid_n, yield_tension):
        self.L = height
        self.initial_conc = initial_conc
        self.particle_diam = particle_diam
        self.particle_esphericity = particle_esphericity
        self.solid_density = solid_density
        self.fluid_density = fluid_density
        self.max_conc = max_conc
        self.M = powerLawFluid_M
        self.n = powerLawFluid_n
        self.yield_tension = yield_tension

cdef class NumericalParameters:
    cdef int N_len
    cdef double total_time
    cdef double timestep
    cdef double maxResidual
    
    def __init__(self, z_divs,total_time,timestep,maxResidual):
        self.N_len = z_divs
        self.total_time = total_time
        self.timestep = timestep
        self.maxResidual = maxResidual

DTYPE = np.double

cdef double perm(double concentration,PhysicalParameters physicalParameters, ConstantParameters constantParameters):
    cdef double permeability
    permeability = (physicalParameters.particle_esphericity * physicalParameters.particle_diam) ** 2 * (1 - concentration) ** 3 / (36 * constantParameters.B * concentration ** 2)
    if permeability < 0:
        return 0
    return permeability

cdef double esp(double x):
    cdef double esphericity
    esphericity = -3.45 * x ** 2 + 5.25 * x - 1.41
    return esphericity

cdef double press_grad(double concentration,ConstantParameters constantParameters):
    cdef double aux, pressure_gradient
    if concentration <= 0:
        return 0
    
    aux = -constantParameters.beta * (constantParameters.ref_conc - concentration) / (concentration * constantParameters.ref_conc)
    pressure_gradient = (constantParameters.p_ref * constantParameters.beta / concentration ** 2) * exp(aux) #Trocar o exp

    return pressure_gradient

cdef double vel(np.ndarray Concentration, int index,PhysicalParameters physicalParameters, ConstantParameters constantParameters,double esph,double rho_susp, double conc_grad):
    cdef double K, aux1, aux2, aux3, aux4, aux5, aux6, velocity

    K = perm(Concentration[index + 1],physicalParameters,constantParameters)
    aux4 = - press_grad(Concentration[index + 1],constantParameters) * conc_grad
    aux4 = press_grad(Concentration[index + 1],constantParameters) * conc_grad
    aux1 = K / (physicalParameters.M * (1 - Concentration[index + 1]) ** (1 - physicalParameters.n))
    aux2 = (physicalParameters.particle_diam / esph) ** (physicalParameters.n - 1) * (rho_susp / (rho_susp -  physicalParameters.solid_density * physicalParameters.initial_conc))
    aux3 = Concentration[index + 1] * (physicalParameters.solid_density - physicalParameters.fluid_density) * (-9.81)
    

    aux5 = aux1 * aux2 * (aux3 - aux4)
    
    if aux5 == 0:
        return 0
    elif aux5 <= 0:
        aux5 = -aux5
    aux6 = 1 / physicalParameters.n
    velocity = pow(aux5, aux6)
    return -velocity

cdef double conc_grad(np.ndarray Concentration,int index,int N_len,double L):
    cdef double concentration_gradient

    concentration_gradient = (Concentration[index + 1] - Concentration[index]) / (L / N_len)

    return concentration_gradient

cdef evalMassConservation(double initial_conc, double solid_density, double length, int n_divs, np.ndarray Concentration):
    cdef double initialmassPerArea, massPerArea = 0
    initialmassPerArea = solid_density * length * n_divs * initial_conc
    for local_concentration in Concentration:
        massPerArea += solid_density * length * local_concentration

    print("\nInitial mass per area: " + str(initialmassPerArea) 
    + " [Kg/m2]\nCurrent total mass per area: " + str(massPerArea) 
    + " [Kg/m2]\nTotal variation (loss): " + str(initialmassPerArea - massPerArea) + " [Kg/m2]")

def RK4Solver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters constantParameters):
    # Parametros do poço 
    cdef double L = physicalParameters.L #5000(m)
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = numericalParameters.N_len #int(L * z_resolution)
    cdef double z_resolution = N_len / L
    cdef double delta_z = L / N_len
    cdef double maxResidual = numericalParameters.maxResidual
    
    # Parametros de sedimentaçao
    cdef double initial_conc = physicalParameters.initial_conc
    cdef double particle_diam = physicalParameters.particle_diam # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef double solid_density = physicalParameters.solid_density
    cdef double fluid_density = physicalParameters.fluid_density
    
    # Parametros de simulaçao
    cdef double total_time = numericalParameters.total_time #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef double timestep = numericalParameters.timestep
    
    # Parametros estimados
    #Permeabilidade
    cdef double max_conc = physicalParameters.max_conc
   
    #Pressao nos solidos
    cdef double beta = constantParameters.beta
    cdef double p_ref = constantParameters.p_ref
    cdef double ref_conc = constantParameters.ref_conc
    
    #Parametros do fluido
    cdef double M = physicalParameters.M
    cdef double n = physicalParameters.n
    
    cdef double esph = esp(physicalParameters.particle_esphericity)
    cdef double mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Inicializacao do data set
    cdef np.ndarray[np.double_t,ndim=1] Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray[np.double_t,ndim=1] Concentration_aux = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray[np.double_t,ndim=1] Velocity = np.zeros(N_len-1, dtype=DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef double currentTime = 0
    
    #Inicializaçao dos vetores de inclinação
    cdef np.ndarray[np.double_t,ndim=1] K1 = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] K2 = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] K3 = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] K4 = np.zeros(N_len, dtype=DTYPE)
    
    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef np.ndarray Pres = np.zeros((N_len, 3660), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, 3660), dtype=DTYPE)
    
    DataToPlot = []
    Time = []
    Time.append(0)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        # Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
        Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)

    print("\n\nData set initialized\n\n")

    cdef int count = 0
    cdef int dia = 0

    cdef int i
    cdef double update

    cdef int nanHit = 0
    cdef int packingIndex = 0

    cdef double concentrationLimit = 0.1395

    cdef double current_tension = 0

    while (currentTime <= total_time):
        #Tempo de relaxamento
        current_tension = physicalParameters.yield_tension * (1 - exp(-currentTime/constantParameters.relax_time))
        # Contagem e gravar dados a cada dia caso a tensao seja maior que a tensao limite
        # if (current_tension - physicalParameters.yield_tension) >= -0.1:
        #     count += 1
        #     if count >= 86400 / timestep:
        #         # print(current_tension)
        #         # print((current_tension - physicalParameters.yield_tension))
        #         # print((current_tension - physicalParameters.yield_tension) <= 0.1)
                
        #         f += 1
        #         for h in range(0,N_len):
        #             Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        #             Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)

        #         dia += 1
                
        #         #Salvar dados a cada dia de simulaçao
        #         Data.append(np.copy(Concentration))
        #         Time.append(dia)
        #         pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresRK4Dia" + str(dia) + ".csv")
        #         pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
        #         pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

        #         #Visualizaçao da simulaçao Debug
        #         print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
        #         print(Concentration)
        #         print("Posição da interface de empacotamento: " + str(packingIndex))
        #         print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
        #         print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
        #         evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
        #         print('Skipped')
        #         count = 0

        #     currentTime += timestep
        #     continue

        #Calculo do vetor de velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration, i, N_len, L)

            # Identificar o valor do gradiente e limitar os saltos de descontinuidade
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Clarificado
                
        #Calculo da inclinaçao K1
        for i in range(0,N_len):
            if i == 0:
                update = - (Concentration[i+1] * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = (Concentration[i] * Velocity[i - 1]) / delta_z
            else:
                update = - (Concentration[i+1] * Velocity[i] - Concentration[i] * Velocity[i - 1]) / delta_z
            K1[i] = update
            Concentration_aux[i] += timestep * update / 2
            if Concentration_aux[i] < 0:
                Concentration_aux[i] = 0

        #Atualizaçao do vetor de velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçao K2
        for i in range(0,N_len):
            if i == 0:
                update = - ((Concentration[i+1] + timestep * K1[i+1] / 2) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K1[i] / 2) * Velocity[i - 1]) / delta_z
            else:
                update = - ((Concentration[i+1] + timestep * K1[i+1] / 2) * Velocity[i] - (Concentration[i] + timestep * K1[i] / 2) * Velocity[i - 1]) / delta_z
            K2[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update / 2
            if Concentration_aux[i] < 0:
                Concentration_aux[i] = 0

        #Atualizaçao da velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçao K3
        for i in range(0,N_len):
            if i == 0:
                update = - ((Concentration[i+1] + timestep * K2[i+1] / 2) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K2[i] / 2) * Velocity[i - 1]) / delta_z
            else:
                update = - ((Concentration[i+1] + timestep * K2[i+1] / 2) * Velocity[i] - (Concentration[i] + timestep * K2[i] / 2) * Velocity[i - 1]) / delta_z
            K3[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update
            if Concentration_aux[i] < 0:
                Concentration_aux[i] = 0

        #Atualizaçao da velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçaio K4
        for i in range(0,N_len):
            if i == 0:
                update = - ((Concentration[i+1] + timestep * K3[i+1]) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K3[i]) * Velocity[i - 1]) / delta_z
            else:
                update = - ((Concentration[i+1] + timestep * K3[i+1]) * Velocity[i] - (Concentration[i] + timestep * K3[i]) * Velocity[i - 1]) / delta_z
            K4[i] = update

            #Verificaçao dos valores de concentraçao
            if np.isnan(update) and nanHit == 0:
                print("Last timestep before fail:")
                print(Concentration)
                nanHit = 1

            Concentration_aux[i] = Concentration[i] + timestep * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) / 6  #Pode ser otimizado excluido a variavel K4 e utilizando o valor de update para o calculo da inclinação media
            if Concentration_aux[i] < 0:
                Concentration_aux[i] = 0
        Concentration = np.copy(Concentration_aux)

        count += 1

        #Atualizar linha de empacotamento
        for i in range(0, N_len):
            if Concentration[i+1] > concentrationLimit:
                packingIndex = i + 1
                continue
            else:
                packingIndex = i
                break

        # Contagem e gravar dados a cada dia
        if count >= 86400 / timestep:
            print(current_tension)

            f += 1
            for h in range(0,N_len):
                Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)

            dia += 1
            
            #Salvar dados a cada dia de simulaçao
            Data.append(np.copy(Concentration))
            Time.append(dia)
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresRK4Dia" + str(dia) + ".csv")
            pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
            pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

            #Visualizaçao da simulaçao Debug
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            print(Concentration)
            print("Posição da interface de empacotamento: " + str(packingIndex))
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            count = 0
        currentTime += timestep
        # Time.append(currentTime)

    for concentrationData in Data:
        DataToPlot.append(concentrationData[0])

    plt.plot(Time,DataToPlot)
    plt.xlabel('Time [Days]')
    plt.ylabel('Concentration')
    plt.savefig('MVF/temporaryFiles/Concentration.png')
    plt.close()

    pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
    pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

    return Data