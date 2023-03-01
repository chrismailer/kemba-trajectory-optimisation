from sympy import Matrix


class Body():
    l = 0.522 # link length [m]
    cg = 0.261 # cg position [m]
    m = 2.886 # mass [kg]
    I = 123624454.40e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


class Femur():
    l = 0.242 # link length [m]
    lp = 0.031 # piston offset from hip [m]
    cg = 0.095 # cg position [m]
    μ = cg/l # cg position as ratio
    m = 0.335 # mass [kg]
    I = 2981115.38e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


class Tibia():
    l = 0.234 # link length [m] 0.254
    lp = 0.048 # piston offset from knee [m]
    cg = 0.109 # cg position [m]
    μ = cg/l # cg position as ratio
    m = 0.159 # mass [kg]
    I = 1264673.14e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


class PistonBody():
    cg = 0.0752 # cg position [m]
    m = 0.230 # mass [kg]
    I = 777104.29e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])

#NB: use 
class PistonRod():
    cg = 0.0825 # cg position [m]
    m = 0.071 # mass [kg]
    I = 225318.30e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


class Rotor():
    N = 6
    m = 0
    I = 12e-5 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


class Boom():
    lx = 2.575 # [m]
    ly = 2.493 # [m]
    y0 = 0.101 # pitch offset [m]
    cg = 1.149 # cg position [m]
    μx = cg/lx
    μy = cg/ly
    m = 2.699 # mass [kg]
    I = 1947254622.19e-9 # inertia [kgm^2]
    M = Matrix([[m,m,I]])


if __name__ == '__main__':
    mass = Boom.m + Body.m + 2*(Femur.m + Tibia.m + PistonBody.m + PistonRod.m)
    print(mass)
