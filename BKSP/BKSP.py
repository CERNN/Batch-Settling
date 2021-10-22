import DataEntry as DataEntry
import ModesBKSP as modes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Parameters: 
    def __init__(self, initialConcentration, maxConcentration, height):
        self.initialConcentration = initialConcentration
        self.height = height
        self.maxConcentration = maxConcentration


class InterfaceTrack:
    def __init__(self,left_concentration,right_concentration,data):
        self.left_concentration = left_concentration
        self.right_concentration = right_concentration
        self.data = data

    def OleinikCriteria(self):
        self.middlePoint = (self.left_concentration + self.right_concentration) / 2
        self.f_left = self.data.interpolate(self.left_concentration)
        self.f_right = data.interpolate(self.right_concentration)
        self.f_middle = data.interpolate(self.middlePoint)
        boundary = (self.f_right - self.f_left) / (self.right_concentration - self.left_concentration)
        lower_boundary = (self.f_middle - self.f_left) / (self.right_concentration - self.left_concentration)
        upper_boundary = (self.f_middle - self.f_right) / (self.left_concentration - self.right_concentration)

        return lower_boundary <= boundary and upper_boundary >= boundary

    def DescreaseRightLimit(self,increment):
        self.right_concentration -= increment

    def DecreaseLeftLimit(self,increment):
        self.left_concentration -= increment

    def IncreaseLeftLimit(self,increment):
        self.left_concentration += increment

    def SetLeftLimit(self,concentration):
        self.left_concentration = concentration

    def SetRightLimit(self,concentration):
        self.right_concentration = concentration

dataSheet = pd.read_csv("BKSP/Dataset3.csv")
data = DataEntry.DataModel(dataSheet)
parameters = Parameters(0.5,0.641996703,9.4)  

modes = {MBS_1: True, MBS-2: True,MBS_3 : True,MBS_4 : True,MBS_5 : True,MBS_6 : True,MBS_7 : True}

upperInterface = InterfaceTrack(0,parameters.initialConcentration,data)
lowerInterface = InterfaceTrack(parameters.initialConcentration,parameters.maxConcentration,data)

descSlope1 = 0
descSlope2 = 0
ascSlope1 = 0 #Implementar função de derivada
ascSlope2 = 0

if upperInterface.OleinikCriteria():
    #MBS 1-5
    plt.plot([upperInterface.left_concentration,upperInterface.right_concentration],[upperInterface.f_left, upperInterface.f_right])
    descSlope1 = upperInterface.f_right / upperInterface.right_concentration
    if lowerInterface.OleinikCriteria() or True: #Manual override
        #MBS 1
        plt.plot([parameters.initialConcentration,parameters.maxConcentration],[lowerInterface.f_left, lowerInterface.f_right])
        ascSlope1 = (lowerInterface.f_right - lowerInterface.f_left) / (lowerInterface.right_concentration - lowerInterface.left_concentration)
    else:
        hasFoundTan = False
        while(not hasFoundTan):
            if lowerInterface.right_concentration > lowerInterface.left_concentration + 0.01:
                lowerInterface.IncreaseLeftLimit(0.01)
            else: 
                #MBS 2-3
                hasFoundTan = True
                f_0 = data.interpolate(lowerInterface.right_concentration)
                f_1 = data.interpolate(lowerInterface.right_concentration - 0.01)
                plt.plot([lowerInterface.right_concentration,lowerInterface.right_concentration - 0.1],[f_0,f_1])
                ascSlope2 = (f_1 - f_0) / 0.01
            if lowerInterface.OleinikCriteria():
                #MBS 4-5
                hasFoundTan = True
                plt.plot([lowerInterface.left_concentration,lowerInterface.right_concentration],[lowerInterface.f_left, lowerInterface.f_right])
                ascSlope2 = (lowerInterface.f_right - lowerInterface.f_left) / (lowerInterface.right_concentration - lowerInterface.left_concentration)

        lowerInterface.SetLeftLimit(parameters.initialConcentration)
        hasFoundTan = False
        while(not hasFoundTan):
            if lowerInterface.right_concentration > lowerInterface.left_concentration + 0.01:
              lowerInterface.DescreaseRightLimit(0.01)
            else:
                #MBS 3-5
                hasFoundTan = True
                f_0 = data.interpolate(lowerInterface.left_concentration)
                f_1 = 2*data.interpolate(lowerInterface.left_concentration + 0.01) - 2*f_0
                plt.plot([lowerInterface.left_concentration,lowerInterface.left_concentration + 0.1],[f_0,f_1])
                ascSlope1 = (f_1 - f_0) / 0.01
            if lowerInterface.OleinikCriteria():
                #MBS 2-4
                hasFoundTan = True
                plt.plot([lowerInterface.left_concentration,lowerInterface.right_concentration],[lowerInterface.f_left, lowerInterface.f_right])
                ascSlope1 = (lowerInterface.f_right - lowerInterface.f_left) / (lowerInterface.right_concentration - lowerInterface.left_concentration)

else:
    #MBS 6-7
    hasFoundTan = False
    while (not hasFoundTan):
        if upperInterface.right_concentration > upperInterface.left_concentration + 0.01:
            upperInterface.DescreaseRightLimit(0.01)
        else:
            #MBS does not exist
            hasFoundTan = True
            f_0 = data.interpolate(upperInterface.left_concentration)
            f_1 = data.interpolate(upperInterface.left_concentration + 0.01)
            plt.plot([upperInterface.left_concentration,upperInterface.left_concentration + 0.01],[f_0,f_1])
            descSlope1 = (f_1 - f_0) / 0.01
        if upperInterface.OleinikCriteria():
            #MBS 6-7
            hasFoundTan = True
            plt.plot([upperInterface.left_concentration,upperInterface.right_concentration],[upperInterface.f_left,upperInterface.f_right])
            descSlope1 = (upperInterface.f_right - upperInterface.f_left) / (upperInterface.right_concentration - upperInterface.left_concentration)

    upperInterface.SetRightLimit(parameters.initialConcentration)
    hasFoundTan = False
    while (not hasFoundTan):
        if upperInterface.right_concentration > upperInterface.left_concentration + 0.01:
            upperInterface.IncreaseLeftLimit(0.01)
        else:
            #MBS 6
            hasFoundTan = True
            f_0 = data.interpolate(upperInterface.right_concentration)
            f_1 = data.interpolate(upperInterface.right_concentration - 0.01)
            plt.plot([upperInterface.right_concentration, upperInterface.right_concentration - 0.01],[f_0, f_1])
            descSlope2 = (f_0 - f_1) / 0.01
        if upperInterface.OleinikCriteria():
            #MBS 7
            hasFoundTan = True
            plt.plot([upperInterface.left_concentration,upperInterface.right_concentration],[upperInterface.f_left, upperInterface.f_right])
            descSlope2 = (upperInterface.f_right - upperInterface.f_left) / (upperInterface.right_concentration - upperInterface.left_concentration)

#MBS 1 plotting
print(lowerInterface.left_concentration)
plt.plot(data.concentrationData,data.fluxData)
plt.xlim(left=0)

plt.ylim(0,-0.03)
plt.show()

t1 = parameters.height / (ascSlope1 - descSlope1)
z1 = descSlope1 * t1 + parameters.height
plt.plot([0,t1],[parameters.height,z1])
plt.plot([0,t1],[0,z1])

# resolution = 8000
# increment = (parameters.maxConcentration - parameters.initialConcentration) / resolution

# lowerInterface.SetLeftLimit(parameters.initialConcentration)
# lowerInterface.SetRightLimit(parameters.maxConcentration)

# zi = z1
# ti = t1

# for i in range(1):
#     lowerInterface.IncreaseLeftLimit(increment)
#     f_0 = data.interpolate(lowerInterface.left_concentration)
#     alpha1 = f_0 / lowerInterface.left_concentration
#     f_0i = data.interpolate(lowerInterface.left_concentration + 0.01)
#     alpha2 = (f_0i - f_0) / 0.01
#     tii = (zi - alpha1 * ti) / (alpha2 - alpha1)
#     zii = alpha2 * ti
#     plt.plot([ti,tii],[zi,zii])
#     ti = tii
#     zi = zii


plt.show()
print(t1)
print(z1)