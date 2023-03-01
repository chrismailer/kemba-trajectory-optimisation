import os
import cloudpickle
import numpy as np
import sympy as sp
from .constants import Body, Femur, Tibia, PistonBody, PistonRod, Rotor, Boom
from pyomo.environ import sin, cos

# TODO
# - check hip angle limit with body

sp.init_printing(use_latex=True)
use_saved_eom = False

x, y, θ = 0, 1, 2 # indexes
g = 9.81

BW = (Boom.m + Body.m + 2*(Femur.m + Tibia.m + PistonBody.m + PistonRod.m)) * g

# generalized coordinates
x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6 = sp.symbols('x_0 y_0 theta_0 theta_1 theta_2 theta_3 theta_4 theta_5 l_5 theta_6 l_6') # position
ẋ0,ẏ0,θ̇0,θ̇1,θ̇2,θ̇3,θ̇4,θ̇5,l̇5,θ̇6,l̇6 = sp.symbols('xdot_0 ydot_0 thetadot_0 thetadot_1 thetadot_2 thetadot_3 thetadot_4 thetadot_5 ldot_5 thetadot_6 ldot_6') # velocity
ẍ0,ÿ0,θ̈0,θ̈1,θ̈2,θ̈3,θ̈4,θ̈5,l̈5,θ̈6,l̈6 = sp.symbols('xddot_0 yddot_0 thetaddot_0 thetaddot_1 thetaddot_2 thetaddot_3 thetaddot_4 thetaddot_5 lddot_5 thetaddot_6 lddot_6') # acceleration

q = sp.Matrix([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6]) # group into matrices
q̇ = sp.Matrix([ẋ0,ẏ0,θ̇0,θ̇1,θ̇2,θ̇3,θ̇4,θ̇5,l̇5,θ̇6,l̇6])
q̈ = sp.Matrix([ẍ0,ÿ0,θ̈0,θ̈1,θ̈2,θ̈3,θ̈4,θ̈5,l̈5,θ̈6,l̈6])

# forces
τf,τb = sp.symbols('tau_f tau_b') # 
Fpf,Fpb = sp.symbols('F_pf F_pb') # piston forces
Fcxf, Fcyf, Fcxb, Fcyb = sp.symbols('F_cx5 F_cy5 F_cx6 F_cy6') # piston connection forces
GRFxf,GRFyf,GRFxb,GRFyb = sp.symbols('GRF_xf GRF_yf GRF_xb GRF_yb')

# useful [x,y] points
p0 = sp.Matrix([x0,y0]) # center of body / end of support pole
p3 = p0 + 0.5*Body.l*sp.Matrix([sp.cos(θ0), sp.sin(θ0)]) # front hip
p4 = p0 - 0.5*Body.l*sp.Matrix([sp.cos(θ0), sp.sin(θ0)]) # back hip
p2 = p3 + Femur.l*sp.Matrix([sp.cos(θ2), sp.sin(θ2)]) # front knee
p5 = p4 - Femur.l*sp.Matrix([sp.cos(θ3), sp.sin(θ3)]) # back knee
p1 = p2 + Tibia.l*sp.Matrix([sp.cos(θ1), sp.sin(θ1)]) # front foot
p6 = p5 - Tibia.l*sp.Matrix([sp.cos(θ4), sp.sin(θ4)]) # back foot
p7 = p2 + Tibia.lp*sp.Matrix([sp.cos(θ1), sp.sin(θ1)]) # front piston attachment
p8 = p5 - Tibia.lp*sp.Matrix([sp.cos(θ4), sp.sin(θ4)]) # back piston attachment
p15 = p3 + Femur.lp*sp.Matrix([sp.cos(θ2+(np.pi/2)), sp.sin(θ2+(np.pi/2))]) # front piston hip offset
p16 = p4 + Femur.lp*sp.Matrix([sp.cos(θ3+(np.pi/2)), sp.sin(θ3+(np.pi/2))]) # back piston hip offset
p9 = p15 + PistonBody.cg*sp.Matrix([sp.cos(θ5), sp.sin(θ5)]) # front piston body cg
p10 = p9 + l5*sp.Matrix([sp.cos(θ5), sp.sin(θ5)]) # front piston rod cg
p11 = p10 + PistonRod.cg*sp.Matrix([sp.cos(θ5), sp.sin(θ5)]) # front piston end
p12 = p16 + PistonBody.cg*sp.Matrix([sp.cos(θ6), sp.sin(θ6)]) # back piston body cg
p13 = p12 + l6*sp.Matrix([sp.cos(θ6), sp.sin(θ6)]) # back piston rod cg
p14 = p13 + PistonRod.cg*sp.Matrix([sp.cos(θ6), sp.sin(θ6)]) # back piston end

r_foot_f = sp.Matrix([p1[x], p1[y], θ1])
r_foot_b = sp.Matrix([p6[x], p6[y], θ4])
r_pf, r_pb = sp.Matrix([p11[x], p11[y], θ5]), sp.Matrix([p14[x], p14[y], θ6]) # piston end
r_af, r_ab = sp.Matrix([p7[x], p7[y], θ1]), sp.Matrix([p8[x], p8[y], θ4]) # attachment point on tibia

# CG of links wrt world frame in generalized coordinates
# ri = [x,y,θ]
r0 = sp.Matrix([x0,y0,θ0]) # body cg
r1 = sp.Matrix([(p1*Tibia.μ+p2*(1-Tibia.μ))[x], (p1*Tibia.μ+p2*(1-Tibia.μ))[y], θ1]) # front tibia cg
rrf = sp.Matrix([p3[x], p3[y], θ0+Rotor.N*(θ2-θ0)]) # front hip rotor
rrb = sp.Matrix([p4[x], p4[y], θ0+Rotor.N*(θ3-θ0)]) # back hip rotor
r2 = sp.Matrix([(p2*Femur.μ+p3*(1-Femur.μ))[x], (p2*Femur.μ+p3*(1-Femur.μ))[y], θ2]) # front femur cg
r3 = sp.Matrix([(p4*(1-Femur.μ)+p5*Femur.μ)[x], (p4*(1-Femur.μ)+p5*Femur.μ)[y], θ3]) # back femur cg
r4 = sp.Matrix([(p5*(1-Tibia.μ)+p6*Tibia.μ)[x], (p5*(1-Tibia.μ)+p6*Femur.μ)[y], θ4]) # back tibia cg
r5 = sp.Matrix([p9[x], p9[y], θ5]) # front piston body cg
r6 = sp.Matrix([p10[x], p10[y], θ5]) # front pison rod cg
r7 = sp.Matrix([p12[x], p12[y], θ6]) # back piston body cg
r8 = sp.Matrix([p13[x], p13[y], θ6]) # back piston rod cg

# velocity of CG in generalized coordinates
ṙ0,ṙ1,ṙ2,ṙ3,ṙ4,ṙ5,ṙ6,ṙ7,ṙ8 = r0.jacobian(q)*q̇, r1.jacobian(q)*q̇, r2.jacobian(q)*q̇, r3.jacobian(q)*q̇, r4.jacobian(q)*q̇, r5.jacobian(q)*q̇, r6.jacobian(q)*q̇, r7.jacobian(q)*q̇, r8.jacobian(q)*q̇
ṙ_foot_f, ṙ_foot_b = r_foot_f.jacobian(q)*q̇, r_foot_b.jacobian(q)*q̇
ṙrf, ṙrb = rrf.jacobian(q)*q̇, rrb.jacobian(q)*q̇

# kinetic energy
# Mb, Mf, Mt, Mpb, Mpr = sp.Matrix([[mb,mb,Ib]]), sp.Matrix([[mf,mf,If]]), sp.Matrix([[mt,mt,It]]), sp.Matrix([[mp,mp,Ip]]), sp.Matrix([[mr,mr,Ir]])
# Mm = sp.Matrix([[0,0,Im]]) # rotor inertia
T = sp.zeros(1)
for M, ṙ in [(Body.M,ṙ0), (Tibia.M,ṙ1), (Femur.M,ṙ2), (Femur.M,ṙ3), (Tibia.M,ṙ4), (PistonBody.M, ṙ5), (PistonRod.M, ṙ6), (PistonBody.M, ṙ7), (PistonRod.M, ṙ8), (Rotor.M, ṙrf), (Rotor.M, ṙrb)]:
    T += 0.5*M*sp.matrix_multiply_elementwise(ṙ,ṙ)
# add boom kinetic energy
T += 0.5 * Boom.M * sp.Matrix([(Boom.μx**2)*(ẋ0**2), (Boom.μy**2)*(ẏ0**2), (ẋ0/Boom.lx)**2 + (ẏ0/Boom.ly)**2])

# potential energy
# rotors don't add any potential energy
V = 0
for m, r in [(Body.m,r0), (Tibia.m,r1), (Femur.m,r2), (Femur.m,r3), (Tibia.m,r4), (PistonBody.m,r5), (PistonRod.m,r6), (PistonBody.m,r7), (PistonRod.m,r8)]:
    V += m*g*r[y]
# add boom potential energy
V += m*g*Boom.cg*((y0-Boom.y0)/Boom.ly)+Boom.y0

L1 = sp.Matrix([T]).jacobian(q̇).jacobian(q)*q̇ + sp.Matrix([T]).jacobian(q̇).jacobian(q̇)*q̈
L3 = sp.Matrix([T]).jacobian(q).T # partial of T in q
L4 = sp.Matrix([V]).jacobian(q).T # partial of U in q

# generalized forces
# τf_0, τb_0 = sp.Matrix([0,0,-τf]), sp.Matrix([0,0,-τb])
# τf_2, τb_3 = sp.Matrix([0,0,τf]), sp.Matrix([0,0,τb])
τf_r, τb_r = sp.Matrix([0,0,τf/Rotor.N]), sp.Matrix([0,0,τb/Rotor.N])

Fpf_5, Fpf_6 = sp.Matrix([-Fpf*sp.cos(θ5),-Fpf*sp.sin(θ5),0]), sp.Matrix([Fpf*sp.cos(θ5),Fpf*sp.sin(θ5),0])
Fpb_7, Fpb_8 = sp.Matrix([-Fpb*sp.cos(θ6),-Fpb*sp.sin(θ6),0]), sp.Matrix([Fpb*sp.cos(θ6),Fpb*sp.sin(θ6),0])

Fc_pf, Fc_af = -sp.Matrix([Fcxf, Fcyf, 0]), sp.Matrix([Fcxf, Fcyf, 0])
Fc_pb, Fc_ab = -sp.Matrix([Fcxb, Fcyb, 0]), sp.Matrix([Fcxb, Fcyb, 0])

GRF_foot_f, GRF_foot_b = sp.Matrix([GRFxf, GRFyf, 0]), sp.Matrix([GRFxb, GRFyb, 0])

Q_GRF = r_foot_f.jacobian(q).T*GRF_foot_f + r_foot_b.jacobian(q).T*GRF_foot_b
Q_Fc = r_pf.jacobian(q).T*Fc_pf + r_pb.jacobian(q).T*Fc_pb + r_af.jacobian(q).T*Fc_af + r_ab.jacobian(q).T*Fc_ab
Q_Fp = r5.jacobian(q).T*Fpf_5 + r6.jacobian(q).T*Fpf_6 + r7.jacobian(q).T*Fpb_7 + r8.jacobian(q).T*Fpb_8
# Q_τ = r0.jacobian(q).T*(τf_0+τb_0) + r2.jacobian(q).T*τf_2 + r3.jacobian(q).T*τb_3
Q_τ = rrf.jacobian(q).T*τf_r + rrb.jacobian(q).T*τb_r
Q = Q_GRF + Q_Fc + Q_Fp + Q_τ

# mass matrix
M = sp.hessian(T, q̇)

# used saved eom if pickled eom exists and script is being imported
cached_eom = './kemba/cache/dynamics.pkl'
if os.path.isfile(cached_eom) and (__name__ != "__main__"):
    with open(cached_eom, mode='rb') as file:
        print('Loading cached dynamics')
        EOM = cloudpickle.load(file)
else:
    print('Computing dynamics')
    EOM = L1 - L3 + L4 - Q
    print('Simplifying... this can take a while')
    EOM = sp.simplify(EOM)
    # save EOM to file
    print('Caching dynamics...')
    outfile = open(cached_eom,'wb')
    cloudpickle.dump(EOM, outfile)
    outfile.close()
    print('Done computing dynamics')

# lambdifying
func_map = [{'sin':sin, 'cos':cos}, np]

vars = [x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6, ẋ0,ẏ0,θ̇0,θ̇1,θ̇2,θ̇3,θ̇4,θ̇5,l̇5,θ̇6,l̇6, ẍ0,ÿ0,θ̈0,θ̈1,θ̈2,θ̈3,θ̈4,θ̈5,l̈5,θ̈6,l̈6, τf,τb, Fpf,Fpb, Fcxf,Fcyf,Fcxb,Fcyb, GRFxf,GRFxb,GRFyf,GRFyb]
λ_EOM = sp.lambdify(vars, EOM, func_map)

λ_footp_f = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6], sp.simplify(r_foot_f), func_map)
λ_footp_b = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6], sp.simplify(r_foot_b), func_map)

λ_footv_f = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6, ẋ0,ẏ0,θ̇0,θ̇1,θ̇2,θ̇3,θ̇4,θ̇5,l̇5,θ̇6,l̇6], sp.simplify(ṙ_foot_f), func_map)
λ_footv_b = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6, ẋ0,ẏ0,θ̇0,θ̇1,θ̇2,θ̇3,θ̇4,θ̇5,l̇5,θ̇6,l̇6], sp.simplify(ṙ_foot_b), func_map)

points = sp.Matrix([p1.T, p2.T, p3.T, p4.T, p5.T, p6.T, p11.T, p14.T, p15.T, p16.T])
λ_points = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6], sp.simplify(points), func_map)

# variables for connections
piston_connection_front = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6], sp.simplify(p11-p7), func_map)
piston_connection_back = sp.lambdify([x0,y0,θ0,θ1,θ2,θ3,θ4,θ5,l5,θ6,l6], sp.simplify(p14-p8), func_map)
