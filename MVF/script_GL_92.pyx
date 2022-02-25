# cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np
import pandas as pd

cimport numpy as np

cdef extern from "math.h":
    double exp(double)

DTYPE = np.float

cdef float conc_grad(np.ndarray Concentration,int index,int N_len,float L):
    cdef float concentration_gradient
    if index == 0:
        concentration_gradient = (Concentration[index + 1] - Concentration[index]) / (L / N_len)
    elif index == (N_len - 1):
        concentration_gradient = (Concentration[index] - Concentration[index - 1]) / (L / N_len)
    else:
        concentration_gradient = (Concentration[index + 1] - Concentration[index - 1]) / (2 * L / N_len)
    
    return concentration_gradient

cdef float esp(float x):
    cdef float esphericity
    esphericity = -3.45 * x ** 2 + 5.25 * x - 1.41
    return esphericity

cdef float permNewt(float x,float k0,float crit_conc, float neta):
    cdef float permeability
    permeability = k0 * (x / crit_conc) ** (-neta)
    return permeability

cdef float press_gradNewt(float x,float a,float b):
    cdef float pressure_gradient
    if x == 0:
        return 0
    pressure_gradient = a * b * x ** (b - 1) #Trocar o exp

    return pressure_gradient

cdef float velNewt(float x,float diam,float k0,float crit_conc,float neta,float M,float esph,float n,float rho_susp,float rho_s,float rho_f,float initial_conc,float conc_grad, float a, float b):
    cdef float K, aux1, aux2, aux3, aux4, aux5, aux6, velocity
    K = permNewt(x,k0,crit_conc,neta)
    aux1 = K / (M * (1 - x) ** (1 - n))
    aux2 = (diam / esph) ** (n - 1) * (rho_susp / (rho_susp -  rho_s * initial_conc))
    aux3 = x * (rho_s - rho_f) * (9.81)
    aux4 = - press_gradNewt(x,a,b) * conc_grad
    
    aux5 = aux1 * aux2 * (aux3 - aux4)
    
    if aux5 == 0:
        return 0
    aux6 = 1 / n
    #print(aux1,aux2,aux3,aux4,aux5)
    velocity = pow(aux5, aux6)
    if velocity < 0:
        return 0
    return -velocity

cdef evalMassConservation(float initial_conc, float solid_density, float length, int n_divs, np.ndarray Concentration):
    cdef float initialmassPerArea, massPerArea = 0
    initialmassPerArea = solid_density * length * n_divs * initial_conc
    for local_concentration in Concentration:
        massPerArea += solid_density * length * local_concentration

    print("\nInitial mass per area: " + str(initialmassPerArea) 
    + " [Kg/m2]\nCurrent total mass per area: " + str(massPerArea) 
    + " [Kg/m2]\nTotal variation (loss): " + str(initialmassPerArea - massPerArea) + " [Kg/m2]")

def CrankNewtSolver():
    # Parametros do poço 
    cdef float L = 0.24 # (m)
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = 240 #int(L * z_resolution)
    cdef float z_resolution = N_len / L
    cdef float delta_z = L / N_len
    cdef float maxResidual = 0.0000001
    
    # Parametros de sedimentaçao
    cdef float initial_conc = 0.09
    cdef float particle_diam = 0.00019321 # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef float solid_density = 2824
    cdef float fluid_density = 1238.96
    
    # Parametros de simulaçao
    cdef float total_time = 8000 #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef float timestep = 0.1
    
    # Parametros estimados
    #Permeabilidade
    cdef float neta = 0.5
    cdef float k0 = 0.0000000085
    cdef float crit_conc = 0.09
    
    #Pressao nos solidos
    cdef float a = 0.00000412
    cdef float b = 12.37
    
    #Parametros do fluido
    cdef float M = 0.3
    cdef float n = 1
    
    
    cdef float esph = esp(0.79)
    cdef float mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Variáveis auxiliares
    cdef float c = timestep / (2 * delta_z)
    cdef float residual
    cdef float distance
    cdef np.ndarray[np.float_t, ndim=1] Concentration_residual = np.ones(N_len, dtype=DTYPE)
    

    #Inicializacao do data set
    cdef np.ndarray[np.float_t,ndim=1] Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray[np.float_t,ndim=1] Concentration_update = np.copy(Concentration)
    cdef np.ndarray[np.float_t,ndim=1] Velocity = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray[np.float_t,ndim=1] Velocity_update = np.copy(Velocity)
    cdef np.ndarray[np.float_t,ndim=1] Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef double currentTime = 0
    
    #Inicializaçao da matrix tridiagonal
    cdef np.ndarray[np.float_t,ndim=2] MatrixA = np.zeros((N_len,N_len), dtype = DTYPE)
    cdef np.ndarray[np.float_t,ndim=1] VectorB = np.zeros(N_len, dtype = DTYPE)
    print("\n\nData set initialized\n\n")
    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef int count = 0
    cdef int dia = 0

    cdef int resIterations

    cdef int i

    for i in xrange(0,N_len):
            grad = conc_grad(Concentration, i, N_len, L)
            if Concentration[i] == 0:
                Velocity[i] = 0
            else:
                Velocity[i] = velNewt(Concentration[i],particle_diam,k0,crit_conc,neta,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,grad,a,b)
            #if i != 0:
             #   if Concentration[i - 1] >= 0.5:
                   # Velocity[i] = 0

    while (currentTime <= total_time):
        Velocity_update = np.copy(Velocity)
        residual = 1
        resIterations = 0
        while(residual > maxResidual):
            resIterations += 1
            #Arranjo na matriz
            for i in xrange(0, N_len) :
                if i == 0: 
                    VectorB[i] = Concentration[i] - c * Concentration[i+1] * Velocity[i+1]
                    MatrixA[i][i] = 1
                    MatrixA[i][i+1] = c * Velocity_update[i+1]
                elif i == N_len - 1:
                    VectorB[i] = Concentration[i] * (1 + c * Velocity[i])
                    MatrixA[i][i] = (1 - c * Velocity_update[i]) 
                else :
                    VectorB[i] = Concentration[i] - c * (Concentration[i+1] * Velocity[i+1] - Concentration[i] * Velocity[i])
                    MatrixA[i][i] = (1 - c * Velocity_update[i])
                    MatrixA[i][i+1] = c * Velocity_update[i+1]

            #Vetor para analise dos residuos
            Concentration_residual = np.copy(Concentration_update)

            #Back substitution
            for i in reversed(xrange(0,N_len)):
                if i == N_len - 1:
                    Concentration_update[i] = VectorB[i] / MatrixA[i][i]
                else:
                    Concentration_update[i] = (VectorB[i] - MatrixA[i][i+1] * Concentration_update[i+1]) / MatrixA[i][i]
            #print(resIterations,residual)
            #Update velocity
            for i in xrange(0,N_len):
                grad = conc_grad(Concentration_update, i, N_len, L)
 
                if Concentration_update[i] == 0:
                    Velocity_update[i] = 0
                else:
                    Velocity_update[i] = velNewt(Concentration_update[i],particle_diam,k0,crit_conc,neta,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,grad,a,b)
                
               # if i != 0:
                    #if Concentration_update[i - 1] >= 0.5:
                      #  Velocity_update[i] = 0

            
            #Residuals evaluation
            residual = 0
            for i in xrange(0,N_len):
                distance = abs(Concentration_update[i] - Concentration_residual[i]) / Concentration_residual[i] 
                if residual < distance:
                    residual = distance

        count += 1
        #print(resIterations,residual)
        if count>100 / timestep:
            print("\nCurrent time:" + str(currentTime))
            #f += 1
            #for h in range(0,N_len):
                #Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                #Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
            dia += 1
            Data.append(np.copy(Concentration))
            pd.DataFrame(Data).to_csv("MVF/temporaryFiles/resultadosPreliminaresNewtDia" + str(dia) + ".csv")
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))
            evalMassConservation(initial_conc, solid_density, L, N_len, Concentration)

            count = 0
        currentTime += timestep
        Velocity = np.copy(Velocity_update)
        Concentration = np.copy(Concentration_update)
        # Time.append(currentTime)

    Data.append([z_resolution,c,N_len])

    return Data