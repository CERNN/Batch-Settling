# cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
import numpy as np

cimport numpy as np

cdef extern from "math.h":
    double exp(double)

DTYPE = np.float

cdef float perm(float x,float diam,float k0,float delta,float max_conc):
    cdef float permeability
    permeability = k0 * diam ** 2 * (max_conc / x - 1) ** delta
    return permeability

cdef float esp(float x):
    cdef float esphericity
    esphericity = -3.45 * x ** 2 + 5.25 * x - 1.41
    return esphericity

cdef float press_grad(float x,float p_ref,float beta,float x_ref):
    cdef float aux, pressure_gradient
    if x == 0:
        return 0
    
    aux = -beta * (x_ref - x) / (x * x_ref)
    pressure_gradient = (p_ref * beta / x ** 2) * exp(aux) #Trocar o exp

    return pressure_gradient


cdef float vel(float x,float diam,float k0,float delta,float max_conc,float M,esph,n,float rho_susp,float rho_s,float rho_f,float initial_conc,float p_ref,beta,float x_ref,float conc_grad) :
    cdef float K, aux1, aux2, aux3, aux4, aux5, aux6, velocity
    K = perm(x,diam,k0,delta,max_conc)
    aux1 = K / (M * (1 - x) ** (1 - n))
    aux2 = (diam / esph) ** (n - 1) * (rho_susp / (rho_susp -  rho_s * initial_conc))
    aux3 = x * (rho_s - rho_f) * (9.81)
    aux4 = - press_grad(x,p_ref,beta,x_ref) * conc_grad
    
    aux5 = aux1 * aux2 * (aux3 - aux4)
    
    if aux5 == 0:
        return 0
    aux6 = 1 / n
    #print(aux1,aux2,aux3,aux4,aux5)
    velocity = pow(aux5, aux6)
    return -velocity

cdef float conc_grad(np.ndarray Concentration,int index,int N_len,float L):
    cdef float concentration_gradient
    if index == 0:
        concentration_gradient = (Concentration[index + 1] - Concentration[index]) / (L / N_len)
    elif index == (N_len - 1):
        concentration_gradient = (Concentration[index] - Concentration[index - 1]) / (L / N_len)
    else:
        concentration_gradient = (Concentration[index + 1] - Concentration[index - 1]) / (2 * L / N_len)
    
    return concentration_gradient

#Calculo de constantes
def EulerSolver():
     
    # Parametros do poço 
    cdef float L = 0.21 #5000(m)
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = 21#int(L * z_resolution)
    cdef float z_resolution = N_len / L
    
    # Parametros de sedimentaçao
    cdef float initial_conc = 0.1391
    cdef float particle_diam = 0.0000408 # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef float solid_density = 2709 # (kg/m3)
    cdef float fluid_density = 1145 # (kg/m3)
    
    # Parametros de simulaçao
    cdef float total_time = 432000 #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef float timestep = 0.1
    
    # Parametros estimados
    #Permeabilidade
    cdef float delta = 0.58 # Permeabilidade - Rocha (2020)
    cdef float k0 = 27.99 # Permeabilidade - Rocha (2020)
    cdef float max_conc = 0.2
    
    #Pressao nos solidos
    cdef float beta = 0.19 # Pressao nos solidos
    cdef float p_ref = 18.62 # Pressao nos solidos
    cdef float ref_conc = 0.145 #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    
    #Parametros do fluido
    cdef float M = 30.13
    cdef float n = 0.21
    
    
    cdef float esph = esp(1)
    cdef float mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Inicializacao do data set
    cdef np.ndarray Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    # Concentration[0] = max_conc
    # Concentration[N_len - 1] = 0
    cdef np.ndarray Velocity = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef float currentTime = 0
    #Time = [currentTime]

    #Pres = np.zeros((N_len, 365), dtype=float)
    #Perm = np.zeros((N_len, 365), dtype=float)
    #f = 0

    #for h in range(0,N_len):
        #Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        #Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

    Data = []
    Data.append(list(Concentration)) #Talvez precise typar

    cdef int count = 0

    cdef int i

    while (currentTime <= total_time):
        
        for i in range(0,N_len):
            grad = conc_grad(Concentration, i, N_len, L)

            if Concentration[i] == 0 or Concentration[i] == max_conc:
                Velocity[i] = 0
            else:
                Velocity[i] = vel(Concentration[i],particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)
        
        for i in range(0,N_len):
            if i == 0:
                update = - timestep * (Concentration[i+1] * Velocity[i+1]) / (L / N_len)
            elif i == (N_len - 1):
                update = + timestep * (Concentration[i] * Velocity[i]) / (L / N_len)
            else:
                update = - timestep * (Concentration[i+1] * Velocity[i+1] - Concentration[i] * Velocity[i]) / (L / N_len)
            Concentration[i] += update
        
        count += 1
        
        if count>86400 / timestep:
            print("\nCurrent time:" + str(currentTime))
            #f += 1
            #for h in range(0,N_len):
                #Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                #Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
            
            Data.append(list(Concentration))
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))

            count = 0
        currentTime += timestep
        # Time.append(currentTime)


    return Data

def NewtonSolver():
    # Parametros do poço 
    cdef float L = 0.21 #5000(m)
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = 21#int(L * z_resolution)
    cdef float z_resolution = N_len / L
    
    # Parametros de sedimentaçao
    cdef float initial_conc = 0.1391
    cdef float particle_diam = 0.0000408 # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef float solid_density = 2709 # (kg/m3)
    cdef float fluid_density = 1145 # (kg/m3)
    
    # Parametros de simulaçao
    cdef float total_time = 432000 #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef float timestep = 0.1
    
    # Parametros estimados
    #Permeabilidade
    cdef float delta = 0.58 # Permeabilidade - Rocha (2020)
    cdef float k0 = 27.99 # Permeabilidade - Rocha (2020)
    cdef float max_conc = 0.2
    
    #Pressao nos solidos
    cdef float beta = 0.19 # Pressao nos solidos
    cdef float p_ref = 18.62 # Pressao nos solidos
    cdef float ref_conc = 0.145 #concentraçao de referencia entre 14.5 e 16% segundo Rocha (2020)
    
    #Parametros do fluido
    cdef float M = 30.13
    cdef float n = 0.21
    
    
    cdef float esph = esp(1)
    cdef float mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Inicializacao do data set
    cdef np.ndarray Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    # Concentration[0] = max_conc
    # Concentration[N_len - 1] = 0
    cdef np.ndarray Velocity = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef float currentTime = 0
    #Time = [currentTime]

    #Pres = np.zeros((N_len, 365), dtype=float)
    #Perm = np.zeros((N_len, 365), dtype=float)
    #f = 0

    #for h in range(0,N_len):
        #Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
        #Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)

    Data = []
    Data.append(list(Concentration)) #Talvez precise typar

    cdef int count = 0

    cdef int i