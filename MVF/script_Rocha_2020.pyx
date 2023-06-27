# cython: boundscheck=False, wraparound=False, nonecheck=False
# python cythonExtension.py build_ext --inplace
import cython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cimport numpy as np

cdef extern from "math.h":
    double exp(double)

cdef class ConstantParameters:
    cdef double delta
    cdef double k0
    cdef double beta
    cdef double ref_conc
    cdef double p_ref

    def __init__(self, delta, k0, beta, ref_conc, p_ref):
        self.delta = delta
        self.k0 = k0
        self.beta = beta
        self.ref_conc = ref_conc
        self.p_ref = p_ref

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

    def __init__(self, height,initial_conc,particle_diam,particle_esphericity,solid_density,fluid_density,max_conc,powerLawFluid_M,powerLawFluid_n):
        self.L = height
        self.initial_conc = initial_conc
        self.particle_diam = particle_diam
        self.particle_esphericity = particle_esphericity
        self.solid_density = solid_density
        self.fluid_density = fluid_density
        self.max_conc = max_conc
        self.M = powerLawFluid_M
        self.n = powerLawFluid_n

cdef class NumericalParameters:
    cdef int N_len
    cdef double total_time
    cdef double timestep
    cdef double maxResidual
    cdef indexesToPlot
    
    def __init__(self, z_divs,total_time,timestep,maxResidual, indexesToPlot):
        self.N_len = z_divs
        self.total_time = total_time
        self.timestep = timestep
        self.maxResidual = maxResidual
        self.indexesToPlot = indexesToPlot

DTYPE = np.double

# cdef double perm(double x,double diam,double k0,double delta,double max_conc):
#     cdef double permeability
#     permeability = k0 * diam ** 2 * (max_conc / x - 1) ** delta
#     # permeability = k0 * pow(diam, 2) * pow((max_conc / x - 1), delta)
#     return permeability

cdef double perm(double concentration,PhysicalParameters physicalParameters, ConstantParameters constantParameters):
    cdef double permeability
    if concentration <= 0:
        return 0
    elif concentration < 0.000001:
        permeability = constantParameters.k0 * physicalParameters.particle_diam ** 2 * (physicalParameters.max_conc / 0.000001 - 1) ** constantParameters.delta
    else:
        permeability = constantParameters.k0 * physicalParameters.particle_diam ** 2 * (physicalParameters.max_conc / concentration - 1) ** constantParameters.delta
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

    # K = perm(Concentration[index + 1],physicalParameters,constantParameters) # Empacotamento, interpolação a jusante
    K = perm(Concentration[index],physicalParameters,constantParameters) # Clarificado, interpolação a montante
    
    # aux1 = K / (physicalParameters.M * (1 - Concentration[index + 1]) ** (1 - physicalParameters.n)) # Empacotamento, interpolação a jusante
    aux1 = K / (physicalParameters.M * (1 - Concentration[index]) ** (1 - physicalParameters.n)) # Clarificado, interpolação a montante

    aux2 = (physicalParameters.particle_diam / esph) ** (physicalParameters.n - 1) * (rho_susp / (rho_susp -  physicalParameters.solid_density * physicalParameters.initial_conc))
    
    # aux3 = Concentration[index + 1] * (physicalParameters.solid_density - physicalParameters.fluid_density) * (-9.81) # Empacotamento, interpolação a jusante
    aux3 = Concentration[index] * (physicalParameters.solid_density - physicalParameters.fluid_density) * (-9.81) # Clarificado, interpolação a montante
    
    # aux4 = press_grad(Concentration[index + 1],constantParameters) * conc_grad # Empacotamento, interpolação a jusante
    aux4 = press_grad(Concentration[index],constantParameters) * conc_grad # Clarificado, interpolação a montante

    aux5 = aux1 * aux2 * (aux3 - aux4)

    # print("Empuxo: " + str(aux3) + "\nPressure: " + str(aux4))
    if aux5 == 0:
        return 0
    elif aux5 < 0:
        aux5 = -aux5
    aux6 = 1 / physicalParameters.n
    velocity = pow(aux5, aux6)
    return -velocity

cdef double conc_grad(np.ndarray Concentration,int index,int N_len,double L):
    cdef double concentration_gradient

    concentration_gradient = (Concentration[index + 1] - Concentration[index]) / (L / N_len)

    return concentration_gradient

cdef double eval_courant_number(np.ndarray velocity_array, int N_len, double L, double timestep):
    cdef double max_velocity
    cdef double min_velocity
    cdef double element_size

    # element_size = double(N_len) / L
    element_size = L / N_len
    max_velocity = abs(velocity_array.max())
    min_velocity = abs(velocity_array.min())
    aux = min_velocity if min_velocity > max_velocity else max_velocity
    min_velocity = max_velocity if aux == min_velocity else min_velocity
    max_velocity = aux
    print("\nCFL Report:"
    + "\nMax CFL: " + str(timestep * max_velocity / element_size) 
    + "\nMin CFL: " + str(timestep * min_velocity / element_size))

    return timestep * max_velocity / element_size

cdef evalMassConservation(double initial_conc, double solid_density, double length, int n_divs, np.ndarray Concentration):
    cdef double initialmassPerArea, massPerArea = 0
    initialmassPerArea = solid_density * length * n_divs * initial_conc
    for local_concentration in Concentration:
        massPerArea += solid_density * length * local_concentration

    print("\nInitial mass per area: " + str(initialmassPerArea) 
    + " [Kg/m2]\nCurrent total mass per area: " + str(massPerArea) 
    + " [Kg/m2]\nTotal variation (loss): " + str(initialmassPerArea - massPerArea) + " [Kg/m2]")


def EulerSolver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters constantParameters):
     
    # Parametros do poço 
    cdef double L = physicalParameters.L
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = numericalParameters.N_len
    cdef double z_resolution = N_len / L
    
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
    cdef double delta = constantParameters.delta
    cdef double k0 = constantParameters.k0
    cdef double max_conc = physicalParameters.max_conc
    
    #Pressao nos solidos
    cdef double beta = constantParameters.beta
    cdef double p_ref = constantParameters.p_ref
    cdef double ref_conc = constantParameters.ref_conc

    #Regiao do clarificado
    cdef clarifiedParameters = constantParameters
    
    #Parametros do fluido
    cdef double M = physicalParameters.M
    cdef double n = physicalParameters.n
    
    
    cdef double esph = esp(physicalParameters.particle_esphericity)
    cdef double mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Inicializacao do data set
    cdef np.ndarray Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray Velocity = np.zeros(N_len - 1, dtype=DTYPE) #Velocidade nas fronteiras do nó
    cdef np.ndarray Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef double currentTime = 0

    Time = []
    Time.append(0)
    
    cdef int days = int(numericalParameters.total_time) / 86400
    cdef np.ndarray Pres = np.zeros((N_len, days), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, days), dtype=DTYPE)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)
        


    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef int count = 0
    cdef int dia = 0
    cdef int i
    cdef int packingIndex = 0

    while (currentTime <= total_time):
        
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration, i, N_len, L)
    
            if Concentration[i + 1] > 0.1395 and Concentration[i] > 0.1395 and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, clarifiedParameters, esph, mixture_density, grad) #Clarificado

            # Velocity[i] = vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad)


        
        for i in range(0,N_len):
            if i == 0:
                update = - timestep * (Concentration[i+1] * Velocity[i]) / (L / N_len)
            elif i == (N_len - 1):
                update = + timestep * (Concentration[i] * Velocity[i - 1]) / (L / N_len)
            else:
                update = - timestep * (Concentration[i+1] * Velocity[i] - Concentration[i] * Velocity[i - 1]) / (L / N_len)
            Concentration[i] += update
        
        count += 1

        #Atualizar linha de empacotamento
        # for i in range(0, N_len):
        #     if Concentration[i+1] > 0.1395:
        #         packingIndex = i + 1
        #         continue
        #     else:
        #         packingIndex = i
        #         break
        
        if count>86400 / timestep:
            
            f += 1
            for h in range(0,N_len):
                Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                # Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
                Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)
                

            dia += 1

            #Visualizaçao da simulaçao Debug
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            print(Concentration)
            print("Posição da interface de empacotamento: " + str(packingIndex))
            # print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            # print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            # eval_courant_number(Velocity,N_len, L, timestep)
            Time.append(dia)
            Data.append(np.copy(Concentration))
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresEulerDia" + str(dia) + ".csv")
            count = 0
        currentTime += timestep
        
    
    #Gerar o plot de concentraçao
    PlotConcentrationData(numericalParameters.indexesToPlot,Data,Time,physicalParameters.L, numericalParameters.N_len)

    #Gerar o plot de pressao
    PlotPressureData(numericalParameters.indexesToPlot, Pres, Time,physicalParameters.L, numericalParameters.N_len)

    #Gerar o plot de permeabilidade
    PlotPermeabilityData(numericalParameters.indexesToPlot, Perm, Time,physicalParameters.L, numericalParameters.N_len)

    pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoEuler.csv")
    pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeEuler.csv")

    return Data

def CrankSolver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters constantParameters):
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
    cdef double delta = constantParameters.delta
    cdef double k0 = constantParameters.k0
    cdef double max_conc = physicalParameters.max_conc
    
    #Pressao nos solidos
    cdef double beta = constantParameters.beta
    cdef double p_ref = constantParameters.p_ref
    cdef double ref_conc = constantParameters.ref_conc
    
    #Parametros do fluido
    cdef double M = physicalParameters.M
    cdef double n = physicalParameters.n
    
    
    cdef double esph = esp(0.75)
    cdef double mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Variáveis auxiliares
    cdef double c = timestep / (2 * delta_z)
    cdef double residual
    cdef double distance
    cdef np.ndarray[np.double_t, ndim=1] Concentration_residual = np.ones(N_len, dtype=DTYPE)
    

    #Inicializacao do data set
    cdef np.ndarray[np.double_t,ndim=1] Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray[np.double_t,ndim=1] Concentration_update = np.copy(Concentration)
    cdef np.ndarray[np.double_t,ndim=1] Velocity = np.zeros(N_len-1, dtype=DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] Velocity_update = np.copy(Velocity)
    cdef np.ndarray[np.double_t,ndim=1] Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef double currentTime = 0
    
    #Inicializaçao da matrix tridiagonal
    cdef np.ndarray[np.double_t,ndim=2] MatrixA = np.zeros((N_len,N_len), dtype = DTYPE)
    cdef np.ndarray[np.double_t,ndim=1] VectorB = np.zeros(N_len, dtype = DTYPE)
    print("\n\nData set initialized\n\n")
    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef np.ndarray Pres = np.zeros((N_len, 365), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, 365), dtype=DTYPE)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        # Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
        Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)
        

    cdef int count = 0
    cdef int dia = 0

    cdef int resIterations

    cdef int i

    for i in xrange(0,N_len-1):
            grad = conc_grad(Concentration, i, N_len, L)

            # Velocity[i] = vel(Concentration,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)
            vel(Concentration,i,physicalParameters, constantParameters, esph, mixture_density, grad)

    while (currentTime <= total_time):
        Velocity_update = np.copy(Velocity)

        residual = 1
        resIterations = 0
        while(residual > maxResidual):
            resIterations += 1
            #Arranjo na matriz
            for i in xrange(0, N_len) :
                if i == 0: 
                    VectorB[i] = Concentration[i] - c * Concentration[i+1] * Velocity[i]
                    MatrixA[i][i] = 1
                    MatrixA[i][i+1] = c * Velocity_update[i]
                elif i == N_len - 1:
                    VectorB[i] = Concentration[i] * (1 + c * Velocity[i-1])
                    MatrixA[i][i] = (1 - c * Velocity_update[i-1]) 
                else :
                    VectorB[i] = Concentration[i] - c * (Concentration[i+1] * Velocity[i] - Concentration[i] * Velocity[i-1])
                    MatrixA[i][i] = (1 - c * Velocity_update[i-1])
                    MatrixA[i][i+1] = c * Velocity_update[i]

            #Vetor para analise dos residuos
            Concentration_residual = np.copy(Concentration_update)

            #Back substitution
            for i in reversed(xrange(0,N_len)):
                if i == N_len - 1:
                    Concentration_update[i] = VectorB[i] / MatrixA[i][i]
                else:
                    Concentration_update[i] = (VectorB[i] - MatrixA[i][i+1] * Concentration_update[i+1]) / MatrixA[i][i]


            #Update velocity
            for i in xrange(0,N_len-1):
                grad = conc_grad(Concentration_update, i, N_len, L)
                Velocity_update[i] = vel(Concentration_update,i,physicalParameters, constantParameters, esph, mixture_density, grad)
                # Velocity_update[i] = vel(Concentration_update,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

            #Residuals evaluation
            residual = 0
            for i in xrange(0,N_len):
                distance = abs(Concentration_update[i] - Concentration_residual[i]) / Concentration_residual[i] 
                if residual < distance:
                    residual = distance

        count += 1
        #print(resIterations,residual)
        if count>86400 / timestep:
            f += 1
            for h in range(0,N_len):
                Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                # Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
                Perm[h][f] = perm(Concentration[h], physicalParameters, constantParameters)

            dia += 1
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            Data.append(np.copy(Concentration))
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresDia" + str(dia) + ".csv")
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            count = 0
        currentTime += timestep
        Velocity = np.copy(Velocity_update)
        Concentration = np.copy(Concentration_update)
        # Time.append(currentTime)

    Data.append([z_resolution,c,N_len])
    
    pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressao.csv")
    pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidade.csv")

    return Data

def RK4Solver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters packingParameters, ConstantParameters clarifiedParameters, np.ndarray Rocha_exp_data, np.ndarray Rocha_num_data):
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
    cdef double delta = packingParameters.delta
    cdef double k0 = packingParameters.k0
    cdef double max_conc = physicalParameters.max_conc
   
    #Pressao nos solidos
    cdef double beta = packingParameters.beta
    cdef double p_ref = packingParameters.p_ref
    cdef double ref_conc = packingParameters.ref_conc
    
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

    cdef int days = int(numericalParameters.total_time) / 86400
    cdef np.ndarray Pres = np.zeros((N_len, days + 1), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, days + 1), dtype=DTYPE)
    
    # DataToPlot = []
    Time = []
    CFL = []
    Time.append(0)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        Perm[h][f] = perm(Concentration[h], physicalParameters, clarifiedParameters)

    print("\n\nData set initialized\n\n")

    cdef double count = 0
    cdef int dia = 0

    cdef int i
    cdef double update

    cdef int nanHit = 0
    cdef int packingIndex = 0

    cdef double minVelocity = 0
    cdef double maxVelocity = 0

    #plotting variables
    cdef int indexToPlot = 0
    cdef double positionToPlot = 0

    cdef double concentrationLimit = 0.1395
    saveFrame(Position, Concentration, dia, physicalParameters.max_conc)
    while (currentTime <= total_time):
        #Calculo do vetor de velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration, i, N_len, L)

            # Identificar o valor do gradiente e limitar os saltos de descontinuidade
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, packingParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, clarifiedParameters, esph, mixture_density, grad) #Clarificado
                
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
            if Concentration_aux[i] < 0 or np.isnan(Concentration_aux[i]):
                Concentration_aux[i] = 0

        #Atualizaçao do vetor de velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, packingParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, clarifiedParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçao K2
        for i in range(0,N_len):
            if i == 0:
                # update = - ((Concentration[i+1] + timestep * K1[i+1] / 2) * Velocity[i]) / delta_z
                update = - ((Concentration[i+1] + timestep * K1[i] / 2) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K1[i] / 2) * Velocity[i - 1]) / delta_z
            else:
                # update = - ((Concentration[i+1] + timestep * K1[i+1] / 2) * Velocity[i] - (Concentration[i] + timestep * K1[i] / 2) * Velocity[i - 1]) / delta_z
                update = - ((Concentration[i+1] + timestep * K1[i] / 2) * Velocity[i] - (Concentration[i] + timestep * K1[i] / 2) * Velocity[i - 1]) / delta_z
            K2[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update / 2
            if Concentration_aux[i] < 0 or np.isnan(Concentration_aux[i]):
                Concentration_aux[i] = 0

        #Atualizaçao da velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, packingParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, clarifiedParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçao K3
        for i in range(0,N_len):
            if i == 0:
                # update = - ((Concentration[i+1] + timestep * K2[i+1] / 2) * Velocity[i]) / delta_z
                update = - ((Concentration[i+1] + timestep * K2[i] / 2) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K2[i] / 2) * Velocity[i - 1]) / delta_z
            else:
                # update = - ((Concentration[i+1] + timestep * K2[i+1] / 2) * Velocity[i] - (Concentration[i] + timestep * K2[i] / 2) * Velocity[i - 1]) / delta_z
                update = - ((Concentration[i+1] + timestep * K2[i] / 2) * Velocity[i] - (Concentration[i] + timestep * K2[i] / 2) * Velocity[i - 1]) / delta_z
            K3[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update
            if Concentration_aux[i] < 0 or np.isnan(Concentration_aux[i]):
                Concentration_aux[i] = 0

        #Atualizaçao da velocidade
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            if Concentration[i + 1] > concentrationLimit and Concentration[i] > concentrationLimit and i < packingIndex:
                Velocity[i] = vel(Concentration,i,physicalParameters, packingParameters, esph, mixture_density, grad) #Empacotamento
            else:
                Velocity[i] = vel(Concentration,i,physicalParameters, clarifiedParameters, esph, mixture_density, grad) #Clarificado

        #Calculo da inclinaçao K4
        for i in range(0,N_len):
            if i == 0:
                # update = - ((Concentration[i+1] + timestep * K3[i+1]) * Velocity[i]) / delta_z
                update = - ((Concentration[i+1] + timestep * K3[i]) * Velocity[i]) / delta_z
            elif i == (N_len - 1):
                update = ((Concentration[i] + timestep * K3[i]) * Velocity[i - 1]) / delta_z
            else:
                # update = - ((Concentration[i+1] + timestep * K3[i+1]) * Velocity[i] - (Concentration[i] + timestep * K3[i]) * Velocity[i - 1]) / delta_z
                update = - ((Concentration[i+1] + timestep * K3[i]) * Velocity[i] - (Concentration[i] + timestep * K3[i]) * Velocity[i - 1]) / delta_z
            K4[i] = update

            #Verificaçao dos valores de concentraçao
            if np.isnan(update) and nanHit == 0:
                print("Last timestep before fail:")
                print(Concentration)
                nanHit = 1

            Concentration_aux[i] = Concentration[i] + timestep * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) / 6  #Pode ser otimizado excluido a variavel K4 e utilizando o valor de update para o calculo da inclinação media
            if Concentration_aux[i] < 0 or np.isnan(Concentration_aux[i]):
                Concentration_aux[i] = 0
        Concentration = np.copy(Concentration_aux)

        # if Velocity.max() > maxVelocity:
        #     maxVelocity = Velocity.max()
        # if Velocity.min() < minVelocity:
        #     minVelocity = Velocity.min()
        # print("Velocidade maxima: " + str(maxVelocity))
        # print("Velocidade minima: " + str(maxVelocity))
        # print(Velocity)

        count += 1

        #Atualizar concentraçao maxima
        # max_conc = 0.25 - 0.04 * (currentTime/total_time)
        # physicalParameters.max_conc = max_conc

        #Atualizar linha de empacotamento
        # for i in range(0, N_len):
        #     if Concentration[i+1] > concentrationLimit:
        #         packingIndex = i + 1
        #         continue
        #     else:
        #         packingIndex = i
        #         break
      
        # if not(np.isnan(np.sum(Concentration))) and dia == 18 and Concentration[N_len - 2] <= 0.1391:
        #     print("Current time: " + str(currentTime))
        #     print(Concentration)
        #     print("Velocity")
        #     print(Velocity)

        # print("Velocity n-1")
        # print(Velocity[N_len - 2])
        # print("Velocity n-2")
        # print(Velocity[N_len - 3])

        if count >= 86400 / timestep:
            f += 1
            for h in range(0,N_len):
                if Concentration[h] > 0:
                    Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                else:
                    Pres[h][f] = 0
                Perm[h][f] = perm(Concentration[h], physicalParameters, packingParameters)

            dia += 1
            
            #Salvar dados a cada dia de simulaçao
            Data.append(np.copy(Concentration))
            Time.append(dia)
            # pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresRK4Dia" + str(dia) + ".csv")
            # pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
            # pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

            # saveFrame(Position, Concentration, dia, physicalParameters.max_conc)

            #Visualizaçao da simulaçao Debug
            print('MaxConc: ' + str(max_conc))
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            print(Concentration)
            # print("Posição da interface de empacotamento: " + str(packingIndex))
            # print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            # print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            max_CFL = eval_courant_number(Velocity,N_len, L, timestep)
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            CFL.append(max_CFL)
            count = (currentTime - dia * 86400) / timestep
        currentTime += timestep
        # Time.append(currentTime)

    print(CFL)
    print("Max CFL: " + str(max(CFL)))

    #Gerar o plot de concentraçao
    PlotConcentrationData(numericalParameters.indexesToPlot,Data,Time,physicalParameters.L, numericalParameters.N_len, physicalParameters.max_conc, num_data=Rocha_num_data, exp_data=Rocha_exp_data)

    # if Pres[days] == 0:
    # if len(Time) != len(Pres[0]):
    #     offset = 1
    #     while (len(Time) < len(Pres[0])):
    #         Time.append(dia+offset)
    #         offset += 1

    pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
    pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

    evaluateConvergence(Concentration=Concentration, init_conc=physicalParameters.initial_conc)

    # #Gerar o plot de pressao
    PlotPressureData(numericalParameters.indexesToPlot, Pres, Time,physicalParameters.L, numericalParameters.N_len)

    # #Gerar o plot de permeabilidade
    PlotPermeabilityData(numericalParameters.indexesToPlot, Perm, Time,physicalParameters.L, numericalParameters.N_len)

    return Data

def PlotConcentrationData(indexesToPlot, Data, Time, L, N_len, max_concentration, np.ndarray num_data, np.ndarray exp_data):
    DataToPlot = []
    colors = ['gray','blue','magenta','red','cyan','green','purple']
    counter = 0
    for index in indexesToPlot:
        DataToPlot = []
        positionToPlot = L * (1 + 2 * index) / (2 * N_len)
        for concentrationData in Data:
            DataToPlot.append(concentrationData[index])
    
        plt.plot(Time,DataToPlot, color=colors[counter], label= "n=" + str(index) + ", z=" + str("{:.2f}".format(positionToPlot * 100)) + " cm")
        plt.legend()
        counter += 1
        # plt.legend()

    DataToPlot = []

    #Dados numericos
    # print(num_data)
    # print(num_data.size)
    if num_data.size != 0:
        plt.plot(num_data[:,0],num_data[:,1], color=colors[0], label='z=0.5cm, Rocha (2020) - Num', linestyle='dashed')
        plt.plot(num_data[:,2],num_data[:,3], color=colors[1], label='z=2.0cm, Rocha (2020) - Num', linestyle='dashed')
        plt.plot(num_data[:,4],num_data[:,5], color=colors[2], label='z=3.0cm, Rocha (2020) - Num', linestyle='dashed')
        # plt.plot(num_data[:,6],num_data[:,7], color=colors[3], label='z=4.0cm, Rocha (2020) - Num', linestyle='dashed')
        # plt.plot(num_data[:,8],num_data[:,9], color=colors[4], label='z=6.0cm, Rocha (2020) - Num', linestyle='dashed')
        # plt.plot(num_data[:,10],num_data[:,11], color=colors[5], label='z=8.0cm, Rocha (2020) - Num', linestyle='dashed')
        # plt.plot(num_data[:,12],num_data[:,13], color=colors[6], label='z=12.0cm, Rocha (2020) - Num', linestyle='dashed')

    #Dados experimentais
    if exp_data.size != 0:
        plt.plot(exp_data[:,0],exp_data[:,1], color=colors[0], label='z=0.5cm, Rocha (2020) - Exp', linestyle='None', marker='s')
        plt.plot(exp_data[:,2],exp_data[:,3], color=colors[1], label='z=2.0cm, Rocha (2020) - Exp', linestyle='None', marker='^')
        plt.plot(exp_data[:,4],exp_data[:,5], color=colors[2], label='z=3.0cm, Rocha (2020) - Exp', linestyle='None', marker='o')
        # plt.plot(exp_data[:,6],exp_data[:,7], color=colors[3], label='z=4.0cm, Rocha (2020) - Exp', linestyle='None', marker='v')
        # plt.plot(exp_data[:,8],exp_data[:,9], color=colors[4], label='z=6.0cm, Rocha (2020) - Exp', linestyle='None', marker='<')
        # plt.plot(exp_data[:,10],exp_data[:,11], color=colors[5], label='z=8.0cm, Rocha (2020) - Exp', linestyle='None', marker='D')
        # plt.plot(exp_data[:,12],exp_data[:,13], color=colors[6], label='z=12.0cm, Rocha (2020) - Exp', linestyle='None', marker='>')

    plt.xlabel('Tempo [Dias]')
    plt.xlim(0.400)
    plt.ylabel('Concentração')
    plt.ylim(0.135,0.22)
    # plt.title('Conc_max = ' + str(max_concentration))
    # plt.legend()
    plt.grid()
    plt.savefig('MVF/temporaryFiles/Concentration.png')
    plt.close()

def PlotPermeabilityData(indexesToPlot, PermeabilityData, Time, L, N_len):
    DataToPlot = []
    for index in indexesToPlot:
        positionToPlot = L * (1 + 2 * index) / (2 * N_len)
        DataToPlot = PermeabilityData[index]

        plt.plot(Time,DataToPlot, label= "n=" + str(index) + ", z=" + str(positionToPlot))
        plt.legend()
    plt.xlabel('Tempo [Dias]')
    plt.ylabel('Permeabilidade')
    # plt.legend()
    plt.savefig('MVF/temporaryFiles/Permeability.png')
    plt.close()

def PlotPressureData(indexesToPlot, PressureData, Time, L, N_len):
    DataToPlot = []
    for index in indexesToPlot:
        positionToPlot = L * (1 + 2 * index) / (2 * N_len)
        DataToPlot = PressureData[index]

        plt.plot(Time,DataToPlot, label= "n=" + str(index) + ", z=" + str(positionToPlot))
        plt.legend()
        # plt.legend()

    plt.xlabel('Tempo [Dias]')
    plt.ylabel('Pressão nos sólidos')
    # plt.legend()
    plt.savefig('MVF/temporaryFiles/Pressure.png')
    plt.close()

def saveFrame(Position, Data, dia, max_conc):

    plt.plot(Position,Data)
    plt.ylim(0,max_conc)
    plt.xlim(0,0.21)
    plt.xlabel('Position [m]')
    plt.ylabel('Concentration')
    plt.title("Dia " + str(dia))
    # plt.legend()
    plt.savefig('MVF/temporaryFiles/animationFrames/Concentration' + str(dia) + '.png')
    plt.close()

def evaluateConvergence(Concentration, init_conc):
    sum = 0
    for node in Concentration:
        sum += (node - init_conc) ** 2
    print("Variação quadratica total: " + str(sum))