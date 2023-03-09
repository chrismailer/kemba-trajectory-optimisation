from math import pi, e
from pyomo.environ import tanh, exp

# T-Motor AK70-10
class AK70_10:
    def __init__(self):
        self.N = 10 # gear ratio
        self.τ_rated = 8.3 # Nm
        self.τ_max = 24.8 # Nm
        self.θ̇_rated = 41.9 # Nm
        self.θ̇_max = 49.7 # rad/s
    
    def torque_limits(self):
        return (-self.τ_max, self.τ_max)
    
    def torque_speed_curve(self, τ, θ̇, dir):
        if dir == '+':
            return θ̇ <= ((self.θ̇_rated - self.θ̇_max)/self.τ_rated) * τ + self.θ̇_max
        else:
            return θ̇ >= ((self.θ̇_rated - self.θ̇_max)/self.τ_rated) * τ - self.θ̇_max

# new actuator to replace AK70-10
class GIM8115:
    def __init__(self):
        self.N = 6 # gear ratio (torque multiplied by this value)
        self.τ_max = 25 # Nm
        self.θ̇_max = 30 # rad/s
        self.I_rotor = 12e-5 # inertia [kgm^2]
    
    def torque_limits(self):
        return (-self.τ_max, self.τ_max)
    
    def torque_speed_curve(self, τ, θ̇, dir):
        if dir == '+':
            return τ <= -(33.6/self.θ̇_max)*θ̇ + 33.6
        else:
            return τ >= -(33.6/self.θ̇_max)*θ̇ - 33.6


# Festo DSNU-25-70
# https://www.festo.com/tw/en/a/1908326/?siteUid=fox_tw&siteName=Festo+TW 
class DSNU_25_70:
    def __init__(self):
        # DSNU-25
        self.stroke = 70e-3 # mm
        self.p = 7e5 # pressure in Pa
        self.Fe_max = 324 # (pi/4)*(0.025**2)*self.p # N
        self.Fr_max = 260 # (pi/4)*(0.025**2 - 0.01**2)*self.p # N
        self.cu = 25
        self.cd = 59
        self.ce = 97
        self.cr = 84
        self.dead_time = 6e-3 # s
    
    def c(self, ue, ur):
        # return self.c_d
        return self.ce*ue*(1-ur) + self.cr*ur*(1-ue) + self.cd*ue*ur + self.cu*(1-ue-ur+ue*ur)
        # https://stackoverflow.com/questions/21293278/mathematical-arithmetic-representation-of-xor
    
    # def static_force(self, ue, ur):
    #     return ue*self.Fe_max - ur*self.Fr_max

    # def model(self, ue, ur, F, dF, ẋ):
    #     # τ = 0.008
    #     τ = 0.015
    #     return τ*dF == self.static_force(ue, ur) - F - ẋ * self.c(ue, ur)
    
    def extend_first_order_model(self, x, ẋ, u, F, dF):
        return F == u*(self.Fe_max) - self.τ(u, x)*dF
    
    def retract_first_order_model(self, x, ẋ, u, F, dF):
        return F == u*(self.Fr_max) - self.τ(u, self.stroke-x)*dF
    
    def τ(self, u, x):
        return (13 - 5*u + (x/self.stroke)*(-7*u + 24))*1e-3
    
    def force(self, ue, ur, F, Fe, Fr, ẋ):
        return F == Fe - Fr - ẋ * self.c(ue,ur)
    
    # implicit euler integration
    def integration(self, F, dF):
        def func(m,fe,cp,j):
            assert cp == 1 and len(m.cp) == 1
            if fe > 1:
                return F[fe,cp,j] == F[fe-1,cp,j] + m.hm*m.h[fe] * dF[fe,cp,j]
            else: # n=1
                return F[fe,cp,j] == 0
        return func



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # plot motor torque speed curve
    motor = GIM8115()
    w = np.linspace(-motor.θ̇_max, motor.θ̇_max, 1000)
    c = 33.6
    m = (-c)/motor.θ̇_max
    τ = np.clip(m*w+c, 0, motor.τ_max)
    plt.plot(w,τ)
    τ = np.clip(m*w-c, -motor.τ_max, 0)
    plt.plot(w,τ)
    plt.show()
