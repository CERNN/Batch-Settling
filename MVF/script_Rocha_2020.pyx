# cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
import pandas as pd

cimport numpy as np

cdef extern from "math.h":
    double exp(double)

DTYPE = np.double

cdef double perm(double x,double diam,double k0,double delta,double max_conc):
    cdef double permeability
    permeability = k0 * diam ** 2 * (max_conc / x - 1) ** delta
    # permeability = k0 * pow(diam, 2) * pow((max_conc / x - 1), delta)
    return permeability

cdef double esp(double x):
    cdef double esphericity
    esphericity = -3.45 * x ** 2 + 5.25 * x - 1.41
    return esphericity

cdef double press_grad(double x,double p_ref,double beta,double x_ref):
    cdef double aux, pressure_gradient
    if x == 0:
        return 0
    
    aux = -beta * (x_ref - x) / (x * x_ref)
    pressure_gradient = (p_ref * beta / x ** 2) * exp(aux) #Trocar o exp

    return pressure_gradient


cdef double vel(np.ndarray Concentration, int index,double diam,double k0,double delta,double max_conc,double M,double esph,double n,double rho_susp,double rho_s,double rho_f,double initial_conc,double p_ref,double beta,double x_ref,double conc_grad):
    cdef double K, aux1, aux2, aux3, aux4, aux5, aux6, velocity, mix_dens

    # Inclusao dos parametros para a regiao do clarificado
    if Concentration[index + 1] < initial_conc: # Testar com index ao inves de index + 1
        K = perm(Concentration[index + 1],diam,52.67,1.04,max_conc)
        aux4 = - press_grad(Concentration[index + 1],3.31,1,x_ref) * conc_grad
    else:
        K = perm(Concentration[index + 1],diam,k0,delta,max_conc)
        aux4 = - press_grad(Concentration[index + 1],p_ref,beta,x_ref) * conc_grad
    # K = perm(Concentration[index + 1],diam,k0,delta,max_conc)
    aux1 = K / (M * (1 - Concentration[index + 1]) ** (1 - n))

    # mix_dens = rho_f + (rho_s - rho_f) * Concentration[index]
    # aux2 = (diam / esph) ** (n - 1) * (mix_dens / (mix_dens -  rho_s * initial_conc))
    aux2 = (diam / esph) ** (n - 1) * (rho_susp / (rho_susp -  rho_s * initial_conc))
    aux3 = Concentration[index + 1] * (rho_s - rho_f) * (-9.81)
    aux4 = press_grad(Concentration[index + 1],p_ref,beta,x_ref) * conc_grad #Erro 1
    # if aux4 != 0:
    #     print(aux4, conc_grad)
    # aux4

    aux5 = aux1 * aux2 * (aux3 - aux4)
    
    if aux5 == 0:
        return 0
    elif aux5 <= 0:
        aux5 = -aux5
    aux6 = 1 / n
    #print(aux1,aux2,aux3,aux4,aux5)
    velocity = pow(aux5, aux6)
    return -velocity

cdef double conc_grad(np.ndarray Concentration,int index,int N_len,double L):
    cdef double concentration_gradient

    concentration_gradient = (Concentration[index + 1] - Concentration[index]) / (L / N_len)

    return concentration_gradient

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
    cdef double solid_density
    cdef double fluid_density
    cdef double max_conc
    cdef double M
    cdef double n

    def __init__(self, height,initial_conc,particle_diam,solid_density,fluid_density,max_conc,powerLawFluid_M,powerLawFluid_n):
        self.L = height
        self.initial_conc = initial_conc
        self.particle_diam = particle_diam
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
    
    def __init__(self, z_divs,total_time,timestep,maxResidual):
        self.N_len = z_divs
        self.total_time = total_time
        self.timestep = timestep
        self.maxResidual = maxResidual


cdef evalMassConservation(double initial_conc, double solid_density, double length, int n_divs, np.ndarray Concentration):
    cdef double initialmassPerArea, massPerArea = 0
    initialmassPerArea = solid_density * length * n_divs * initial_conc
    for local_concentration in Concentration:
        massPerArea += solid_density * length * local_concentration

    print("\nInitial mass per area: " + str(initialmassPerArea) 
    + " [Kg/m2]\nCurrent total mass per area: " + str(massPerArea) 
    + " [Kg/m2]\nTotal variation (loss): " + str(initialmassPerArea - massPerArea) + " [Kg/m2]")

#Calculo de constantes
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
    
    #Parametros do fluido
    cdef double M = physicalParameters.M
    cdef double n = physicalParameters.n
    
    
    cdef double esph = esp(1)
    cdef double mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Inicializacao do data set
    cdef np.ndarray Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray Velocity = np.zeros(N_len - 1, dtype=DTYPE) #Velocidade nas fronteiras do nó
    cdef np.ndarray Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef double currentTime = 0

    #Time = [currentTime]

    cdef np.ndarray Pres = np.zeros((N_len, 365), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, 365), dtype=DTYPE)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef int count = 0
    cdef int dia = 0
    cdef int i

    while (currentTime <= total_time):
        
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration, i, N_len, L)
    
            Velocity[i] = vel(Concentration,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)


        
        for i in range(0,N_len):
            if i == 0:
                update = - timestep * (Concentration[i+1] * Velocity[i]) / (L / N_len)
            elif i == (N_len - 1):
                update = + timestep * (Concentration[i] * Velocity[i - 1]) / (L / N_len)
            else:
                update = - timestep * (Concentration[i+1] * Velocity[i] - Concentration[i] * Velocity[i - 1]) / (L / N_len)
            Concentration[i] += update
        
        count += 1
        
        if count>86400 / timestep:
            
            f += 1
            for h in range(0,N_len):
                Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

            dia += 1
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            Data.append(np.copy(Concentration))
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresEulerDia" + str(dia) + ".csv")
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            count = 0
        currentTime += timestep
        # Time.append(currentTime)

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
        Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

    cdef int count = 0
    cdef int dia = 0

    cdef int resIterations

    cdef int i

    for i in xrange(0,N_len-1):
            grad = conc_grad(Concentration, i, N_len, L)

            Velocity[i] = vel(Concentration,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

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

                Velocity_update[i] = vel(Concentration_update,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

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
                Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

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
    
    
    cdef double esph = esp(0.8)
    cdef double mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    # #Variáveis auxiliares
    # cdef double c = timestep / (2 * delta_z)
    
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



    print("\n\nData set initialized\n\n")
    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef np.ndarray Pres = np.zeros((N_len, 3660), dtype=DTYPE)
    cdef np.ndarray Perm = np.zeros((N_len, 3660), dtype=DTYPE)
    cdef int f = 0

    for h in range(0,N_len):
        Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

    cdef int count = 0
    cdef int dia = 0

    cdef int i
    cdef double update

    while (currentTime <= total_time):
        
        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration, i, N_len, L)

            Velocity[i] = vel(Concentration,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

        for i in range(0,N_len):
            if i == 0:
                update = - (Concentration[i+1] * Velocity[i]) / delta_z
                # print(update)
            elif i == (N_len - 1):
                update = (Concentration[i] * Velocity[i - 1]) / delta_z
            else:
                update = - (Concentration[i+1] * Velocity[i] - Concentration[i] * Velocity[i - 1]) / delta_z
            K1[i] = update
            Concentration_aux[i] += timestep * update / 2


        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            Velocity[i] = vel(Concentration_aux,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)
        

        for i in range(0,N_len):
            if i == 0:
                update = - (Concentration_aux[i+1] * Velocity[i]) / delta_z
                # print(update)
            elif i == (N_len - 1):
                update = (Concentration_aux[i] * Velocity[i - 1]) / delta_z
            else:
                update = - (Concentration_aux[i+1] * Velocity[i] - Concentration_aux[i] * Velocity[i - 1]) / delta_z
            K2[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update / 2

        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            Velocity[i] = vel(Concentration_aux,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

        for i in range(0,N_len):
            if i == 0:
                update = - (Concentration_aux[i+1] * Velocity[i]) / delta_z
                # print(update)
            elif i == (N_len - 1):
                update = (Concentration_aux[i] * Velocity[i - 1]) / delta_z
            else:
                update = - (Concentration_aux[i+1] * Velocity[i] - Concentration_aux[i] * Velocity[i - 1]) / delta_z
            K3[i] = update
            Concentration_aux[i] = Concentration[i] + timestep * update

        for i in range(0,N_len - 1):
            grad = conc_grad(Concentration_aux, i, N_len, L)
    
            Velocity[i] = vel(Concentration_aux,i,particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

        for i in range(0,N_len):
            if i == 0:
                update = - (Concentration_aux[i+1] * Velocity[i]) / delta_z
                # print(update)
                # print((K1[i] + 2 * K2[i] + 2 * K3[i] + update) / 6)
                # print("RK4 ^")
            elif i == (N_len - 1):
                update = (Concentration_aux[i] * Velocity[i - 1]) / delta_z
            else:
                update = - (Concentration_aux[i+1] * Velocity[i] - Concentration_aux[i] * Velocity[i - 1]) / delta_z
            K4[i] = update
            Concentration[i] += timestep * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) / 6 #Pode ser otimizado excluido a variavel K4 e utilizando o valor de update para o calculo da inclinação media

        count += 1
        
        if count>=86400 / timestep:
            
            f += 1
            for h in range(0,N_len):
                Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

            dia += 1
            print("\nCurrent time:" + str(currentTime) + "\nDia: " + str(dia))
            Data.append(np.copy(Concentration))
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresRK4Dia" + str(dia) + ".csv")
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)
            count = 0
        currentTime += timestep
        # Time.append(currentTime)

    pd.DataFrame(Pres).to_csv("MVF/temporaryFiles/resultadosPressaoRK4.csv")
    pd.DataFrame(Perm).to_csv("MVF/temporaryFiles/resultadosPermeabilidadeRK4.csv")

    return Data