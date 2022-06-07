from os import times_result
from time import time
from turtle import position
import numpy as np
import matplotlib.pyplot as plt
import math as math
import random

# Constantes
k = 100
h = 1000
rho = 2300
c = 88
alpha = k / (rho * c)

#Condiçoes de contorno
T0 = 100.0
Tinf = 20.0

# Numerico
currentTime = 0
# total_time = 100000
length = 1
x_divs = 40
numberOfSteps = 10001

delta_x = length / x_divs
square_delta_x = pow(delta_x, 2)
# timestep = 0.1 * square_delta_x / alpha
timestep = 0.1

stability = alpha * timestep * (h * timestep / k + 1) / square_delta_x
print(stability)


total_time = timestep * numberOfSteps

steps = int(total_time/timestep)
n = 0

# Vetores
DTYPE = np.double
T = np.ones((steps, x_divs + 1), dtype=DTYPE)
T[0] *= T0
# T[0][0] = T1
Position = np.arange(x_divs + 1) * delta_x

K1 = np.zeros(x_divs + 1, dtype=DTYPE)
K2 = np.zeros(x_divs + 1, dtype=DTYPE)
K3 = np.zeros(x_divs + 1, dtype=DTYPE)
K4 = np.zeros(x_divs + 1, dtype=DTYPE)

while (n < steps - 1):

    #Calculo da inclinaçao K1
    for i in range(0,x_divs + 1):
        if i == 0:
            update = 0 # At boundary apply euler method
        elif i == x_divs:
            update = 0 # At boundary apply euler method
        else:
            update = (T[n][i+1] - 2 * T[n][i] + T[n][i-1]) / square_delta_x
        K1[i] = update

    #Calculo da inclinaçao K2
    for i in range(0,x_divs + 1):
        if i == 0:
            update = 0
        elif i == x_divs:
            update = 0
        else:
            # update = (T[n][i+1] + timestep * K1[i+1] / 2 - 2 * (T[n][i] + timestep * K1[i] / 2) + T[n][i-1] + timestep * K1[i-1] / 2) / square_delta_x
            update = (T[n][i+1] + timestep * K1[i] / 2 - 2 * (T[n][i] + timestep * K1[i] / 2) + T[n][i-1] + timestep * K1[i] / 2) / square_delta_x
        K2[i] = update

    #Calculo da inclinaçao K3
    for i in range(0,x_divs + 1):
        if i == 0:
            update = 0
        elif i == x_divs:
            update = 0
        else:
            # update = (T[n][i+1] + timestep * K2[i+1] / 2 - 2 * (T[n][i] + timestep * K2[i] / 2) + T[n][i-1] + timestep * K2[i-1] / 2) / square_delta_x
            update = (T[n][i+1] + timestep * K2[i] / 2 - 2 * (T[n][i] + timestep * K2[i] / 2) + T[n][i-1] + timestep * K2[i] / 2) / square_delta_x
        K3[i] = update

    #Calculo da inclinaçaio K4
    for i in range(0,x_divs + 1):
        if i == 0:
            update = 0
        elif i == x_divs:
            update = 0
        else:
            # update = (T[n][i+1] + timestep * K3[i+1] - 2 * (T[n][i] + timestep * K3[i]) + T[n][i-1] + timestep * K3[i-1]) / square_delta_x
            update = (T[n][i+1] + timestep * K3[i] - 2 * (T[n][i] + timestep * K3[i]) + T[n][i-1] + timestep * K3[i]) / square_delta_x
        K4[i] = update

    #Avanço temporal

    for i in range(0,x_divs + 1):
        if i == 0:
            # T[n+1][i] = T[n][i] # Boundary condition, constant temperature
            T[n+1][i] = T[n][i] + 2 * alpha * timestep * (-(T[n][i] - T[n][i+1]) / delta_x + h * (Tinf - T[n][i]) / k) / delta_x # Boundary condition, convection
        elif i == x_divs:
            # T[n+1][i] = T[n][i] - 2 * alpha * timestep * (T[n][i] - T[n][i-1]) / square_delta_x # At boundary apply euler method
            T[n+1][i] = T[n][i] + 2 * alpha * timestep * (-(T[n][i] - T[n][i-1]) / delta_x + h * (Tinf - T[n][i]) / k) / delta_x # Boundary condition, convection
        else:
            # T[n+1][i] = T[n][i] + alpha * timestep * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) / 6
            T[n+1][i] = T[n][i] + alpha * timestep * (T[n][i+1] - 2 * T[n][i] + T[n][i-1]) / square_delta_x # Euler
         
    currentTime += timestep
    n += 1

    #Visualizaçao da simulaçao Debug
    # print("\nCurrent time:" + str(currentTime))
    # print(T[n])
    # print(str(T[n].min()) + " -> " + str(np.where(T == T.min())[0][0]))
    # print(str(T[n].max()) + " -> " + str(np.where(T == T.max())[0][0]))

PlotSteps = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
colors = []
for step in PlotSteps:
    colors.append((random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),random.uniform(0.5,1)))
# PlotSteps = [2000, 5000, 10000]

#Analitic solution
K1 = 0.1
res = 1
Lc = length / 2

while res > 0.0001:
    Bi = h*Lc/(k * K1)
    aux = math.atan(Bi)
    res = abs(K1 - aux) / K1
    K1 = aux



# K2 = 4.3058
# K3 = 7.2281
# K4 = 10.2003

C1 = (4 * math.sin(K1)) / (2 * K1 + math.sin(2 * K1))
# C2 = (4 * math.sin(K2)) / (2 * K2 + math.sin(2 * K2))
# C3 = (4 * math.sin(K3)) / (2 * K3 + math.sin(2 * K3))
# C4 = (4 * math.sin(K4)) / (2 * K4 + math.sin(2 * K4))

print('Bi= ' + str(h*Lc/k))
# print('Fo= ' + str(alpha * total_time / length ** 2))
print(K1)
print(C1)

#plotting
counter = 0
for targetStep in PlotSteps:
    plt.plot(Position, T[targetStep], label ='Numerico, t=' + str(targetStep*timestep) + ' s', color=colors[counter])

    TAnal = []

    timeLocal = targetStep * timestep
    Fo = alpha * timeLocal / pow(Lc,2)

    for index in range(0,x_divs+1):
        print('Fo= ' + str(alpha * timeLocal / length ** 2))
        x = Position[index] - length/2
        x *= 2

        theta_star = C1 * math.exp(-pow(K1, 2) * Fo) * math.cos(K1 * x / length) 
        # + C2 * math.exp(-pow(K2, 2) * Fo) * math.cos(K2 * x / length) 
        # + C3 * math.exp(-pow(K3, 2) * Fo) * math.cos(K3 * x / length) 
        # + C4 * math.exp(-pow(K4, 2) * Fo) * math.cos(K4 * x / length) 


        # t = Tinf + (T0 - Tinf) * C1 * math.exp(-pow(K1, 2) * Fo) * math.cos(K1 * x / length) 
        t = Tinf + (T0 - Tinf) * theta_star
        TAnal.append(t)

    if Fo > 0.2:
        plt.plot(Position, TAnal, linestyle='none', marker='o', label='Analitico, t=' + str(targetStep*timestep), color=colors[counter])
    counter+=1

plt.legend()
# plt.plot(T[100], Position)
# plt.plot(T[steps-1], Position)

# TINF = np.ones(x_divs+1, dtype=DTYPE) * Tinf
# plt.plot(Position, TINF)


plt.ylim(Tinf,T0+1)
plt.xlim(0,length)
plt.ylabel('Temperatura [°C]')
plt.xlabel('Posição [m]')
plt.savefig('MVF/temporaryFiles/Temperature.png')
plt.show()

