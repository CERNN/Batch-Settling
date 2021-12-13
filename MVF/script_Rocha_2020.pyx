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

cdef class ConstantParameters:
    cdef float delta
    cdef float k0
    cdef float beta
    cdef float ref_conc
    cdef float p_ref

    def __init__(self, delta, k0, beta, ref_conc, p_ref):
        self.delta = delta
        self.k0 = k0
        self.beta = beta
        self.ref_conc = ref_conc
        self.p_ref = p_ref

cdef class PhysicalParameters:
    cdef float L
    cdef float initial_conc
    cdef float particle_diam
    cdef float solid_density
    cdef float fluid_density
    cdef float max_conc
    cdef float M
    cdef float n

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
    cdef float total_time
    cdef float timestep
    cdef float maxResidual
    
    def __init__(self, z_divs,total_time,timestep,maxResidual):
        self.N_len = z_divs
        self.total_time = total_time
        self.timestep = timestep
        self.maxResidual = maxResidual


#Calculo de constantes
def EulerSolver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters constantParameters):
     
    # Parametros do poço 
    cdef float L = physicalParameters.L
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = numericalParameters.N_len
    cdef float z_resolution = N_len / L
    
    # Parametros de sedimentaçao
    cdef float initial_conc = physicalParameters.initial_conc
    cdef float particle_diam = physicalParameters.particle_diam # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef float solid_density = physicalParameters.solid_density
    cdef float fluid_density = physicalParameters.fluid_density
    
    # Parametros de simulaçao
    cdef float total_time = numericalParameters.total_time #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef float timestep = numericalParameters.timestep
    
    # Parametros estimados
    #Permeabilidade
    cdef float delta = constantParameters.delta
    cdef float k0 = constantParameters.k0
    cdef float max_conc = physicalParameters.max_conc
    
    #Pressao nos solidos
    cdef float beta = constantParameters.beta
    cdef float p_ref = constantParameters.p_ref
    cdef float ref_conc = constantParameters.ref_conc
    
    #Parametros do fluido
    cdef float M = physicalParameters.M
    cdef float n = physicalParameters.n
    
    
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
    Data.append(np.copy(Concentration)) #Talvez precise typar

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
            
            Data.append(np.copy(Concentration))
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))

            count = 0
        currentTime += timestep
        # Time.append(currentTime)


    return Data

def CrankSolver(PhysicalParameters physicalParameters, NumericalParameters numericalParameters, ConstantParameters constantParameters):
    # Parametros do poço 
    cdef float L = physicalParameters.L #5000(m)
    # z_resolution = 220 #div/m 80 a 100 div/m -> Prof Calcada
    cdef int N_len = numericalParameters.N_len #int(L * z_resolution)
    cdef float z_resolution = N_len / L
    cdef float delta_z = L / N_len
    cdef float maxResidual = numericalParameters.maxResidual
    
    # Parametros de sedimentaçao
    cdef float initial_conc = physicalParameters.initial_conc
    cdef float particle_diam = physicalParameters.particle_diam # (m) D10 - 3.008 microns D50 - 40.803 mic D90 - 232.247 mic -> Usar D50
    cdef float solid_density = physicalParameters.solid_density
    cdef float fluid_density = physicalParameters.fluid_density
    
    # Parametros de simulaçao
    cdef float total_time = numericalParameters.total_time #31536000#(s) 31,536,000 -> um ano / 50 dias - 4,320,000
    cdef float timestep = numericalParameters.timestep
    
    # Parametros estimados
    #Permeabilidade
    cdef float delta = constantParameters.delta
    cdef float k0 = constantParameters.k0
    cdef float max_conc = physicalParameters.max_conc
    
    #Pressao nos solidos
    cdef float beta = constantParameters.beta
    cdef float p_ref = constantParameters.p_ref
    cdef float ref_conc = constantParameters.ref_conc
    
    #Parametros do fluido
    cdef float M = physicalParameters.M
    cdef float n = physicalParameters.n
    
    
    cdef float esph = esp(1)
    cdef float mixture_density = fluid_density + (solid_density - fluid_density) * initial_conc

    #Variáveis auxiliares
    cdef float c = timestep / (2 * delta_z)
    cdef float residual
    cdef float distance
    cdef np.ndarray Concentration_residual = np.ones(N_len, dtype=DTYPE)
    

    #Inicializacao do data set
    cdef np.ndarray Concentration = np.ones(N_len, dtype=DTYPE) * initial_conc
    cdef np.ndarray Concentration_update = np.copy(Concentration)
    # Concentration[0] = max_conc
    # Concentration[N_len - 1] = 0
    cdef np.ndarray Velocity = np.zeros(N_len, dtype=DTYPE)
    cdef np.ndarray Velocity_update = np.copy(Velocity)
    cdef np.ndarray Position = 0.5 / z_resolution + np.arange(N_len, dtype=DTYPE) * 1 / z_resolution
    cdef float currentTime = 0
    
    #Inicializaçao da matrix tridiagonal
    cdef np.ndarray MatrixA = np.zeros((N_len,N_len), dtype = DTYPE)
    cdef np.ndarray VectorB = np.zeros(N_len, dtype = DTYPE)

    Data = []
    Data.append(np.copy(Concentration)) #Talvez precise typar

    cdef int count = 0

    cdef int resIterations

    cdef int i

    for i in xrange(0,N_len):
            grad = conc_grad(Concentration, i, N_len, L)

            if Concentration[i] == 0 or Concentration[i] == max_conc:
                Velocity[i] = 0
            else:
                Velocity[i] = vel(Concentration[i],particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

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


            #Update velocity
            for i in xrange(0,N_len):
                grad = conc_grad(Concentration_update, i, N_len, L)

                if Concentration_update[i] == 0 or Concentration_update[i] == max_conc:
                    Velocity_update[i] = 0
                else:
                    Velocity_update[i] = vel(Concentration_update[i],particle_diam,k0,delta,max_conc,M,esph,n,mixture_density,solid_density,fluid_density,initial_conc,p_ref,beta,ref_conc,grad)

            #Residuals evaluation
            residual = 0
            for i in xrange(0,N_len):
                distance = abs(Concentration_update[i] - Concentration_residual[i]) / Concentration_residual[i] 
                if residual < distance:
                    residual = distance

        count += 1
        #print(resIterations,residual)
        if count>86400 / timestep:
            print("\nCurrent time:" + str(currentTime))
            #f += 1
            #for h in range(0,N_len):
                #Pres[h][f] = p_ref * np.exp(-beta * (1 / Concentration[h] - 1 / ref_conc))
                #Perm[h][f] = perm(Concentration[h], particle_diam, k0, delta, max_conc)
            
            Data.append(np.copy(Concentration))
            print(str(Concentration.min()) + " -> " + str(np.where(Concentration == Concentration.min())[0][0]))
            print(str(Concentration.max()) + " -> " + str(np.where(Concentration == Concentration.max())[0][0]))

            count = 0
        currentTime += timestep
        Velocity = np.copy(Velocity_update)
        Concentration = np.copy(Concentration_update)
        # Time.append(currentTime)

    Data.append([z_resolution,c,N_len])

    return Data