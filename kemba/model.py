from pyomo.environ import*
from core.collocation import implicit_euler
from core.actuators import GIM8115, DSNU_25_70
import numpy as np
import kemba.dynamics as eom

# TODO
# - make nodes or timestep an input, not both
# - change joints to be front and back, hip and knee
# - add cost of transport cost function
# - how to handle piston collocation?

def create_model(duration, nfe, ncp, integrator=implicit_euler, μ=1.0, timestep_bounds=(0.8,1.2)):
    model = ConcreteModel(name='kemba') # create the model

    # scaling factor for forces and torques
    # solver prefers when values are around ±1
    sf = eom.BW

    # Variables
    motor = GIM8115()
    piston = DSNU_25_70()

    # SETS
    model.fe = RangeSet(nfe)
    model.cp = RangeSet(ncp)
    hm0 = duration/nfe
    model.hm = Param(initialize=hm0)
    model.h = Var(model.fe, bounds=timestep_bounds) # usually ±20%
    
    gDOF = ['x0','y0','θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'l5', 'θ6', 'l6'] # generalized degrees of freedom
    x0, y0, θ0 = gDOF.index('x0'), gDOF.index('y0'), gDOF.index('θ0')
    model.gDOF = Set(initialize=gDOF, ordered=True)

    cDOF = ['x','y'] # cartesian degrees of freedom
    model.cDOF = Set(initialize=cDOF, ordered=True)
    x, y = cDOF.index('x'), cDOF.index('y')
    model.R = Set(initialize=cDOF, ordered=True)

    #! PARAMETERS
    gravity = {'x': 0.0, 'y': -9.81}
    model.g = Param(model.cDOF, initialize=gravity)

    #! STATE VARIABLES
    # system coordinates
    model.q = Var(model.fe, model.cp, model.gDOF, bounds=(-100, 100)) # position
    model.dq = Var(model.fe, model.cp, model.gDOF) # velocity
    model.ddq = Var(model.fe, model.cp, model.gDOF) # acceleration

    #! INTEGRATION
    # integration constraints
    model.integrate_q = Constraint(model.fe, model.cp, model.gDOF, rule=integrator(model.q, model.dq))
    model.integrate_dq = Constraint(model.fe, model.cp, model.gDOF, rule=integrator(model.dq, model.ddq))

    #! GROUND INTERACTIONS
    # paramters
    model.μ = Param(initialize=μ) # friction coefficient

    signs = ['+','-'] # sign set for positive and negative components
    model.signs = Set(initialize=signs)

    feet = ['front_foot', 'back_foot']
    model.feet = Set(initialize=feet, ordered=True)

    # contact variables
    model.foot_y = Var(model.fe, model.cp, model.feet, bounds=(0, None))
    model.foot_dx = Var(model.fe, model.cp, model.feet, model.signs, bounds=(0, None))

    model.GRFx = Var(model.fe, model.cp, model.feet, model.signs, bounds=(0, 10))
    model.GRFy = Var(model.fe, model.cp, model.feet, bounds=(0, 10))

    # model.friction_cone = Var(model.fe, model.feet, bounds=(0, None))

    model.contact_penalty = Var(model.fe, model.feet, bounds=(0, 1))
    # model.friction_penalty = Var(model.fe, model.feet, bounds=(0, 1))
    model.slip_penalty = Var(model.fe, model.feet, model.signs, bounds=(0, 1))
    
    # keep hip and knee points above ground
    def solid_ground(m,n,cp,i):
        return eom.λ_points(*m.q[n,cp,:])[i,1] >= 0.03 # [m] knee radius is 16mm
    model.solid_ground = Constraint(model.fe, model.cp, RangeSet(4), rule=solid_ground)

    # constraints: aux variables
    def get_foot_y(m,n,cp,f):
        if (n == 1 and cp < m.cp[-1]):
            return Constraint.Skip
        elif f == 'front_foot':
            vars = m.q[n,cp,:]
            return model.foot_y[n,cp,f] == eom.λ_footp_f(*vars).flat[y]
        elif f == 'back_foot':
            vars = m.q[n,cp,:]
            return m.foot_y[n,cp,f] == eom.λ_footp_b(*vars).flat[y]
        else:
            return Constraint.Skip
    model.def_foot_y = Constraint(model.fe, model.cp, model.feet, rule=get_foot_y)

    def get_foot_dx(m,n,cp,f):
        if (n == 1 and cp < m.cp[-1]):
            return Constraint.Skip
        elif f == 'front_foot':
            vars = [*m.q[n,cp,:], *m.dq[n,cp,:]]
            return m.foot_dx[n,cp,f,'+'] - m.foot_dx[n,cp,f,'-'] == eom.λ_footv_f(*vars).flat[x]
        elif f == 'back_foot':
            vars = [*m.q[n,cp,:], *m.dq[n,cp,:]]
            return m.foot_dx[n,cp,f,'+'] - m.foot_dx[n,cp,f,'-'] == eom.λ_footv_b(*vars).flat[x]
        else:
            return Constraint.Skip
    model.def_foot_dx = Constraint(model.fe, model.cp, model.feet, rule=get_foot_dx)

    # def friction_cone(m,n,f):
    #     return m.friction_cone[n,f] == m.μ * sum(m.GRFy[n,:,f]) - sum(m.GRFx[n,:,f,:])
    # model.def_friction_cone = Constraint(model.fe, model.feet, rule=friction_cone)

    def friction_cone(m,n,f):
        return sum(m.GRFx[n,:,f,:]) <= m.μ * sum(m.GRFy[n,:,f])
    model.friction_cone = Constraint(model.fe, model.feet, rule=friction_cone)

    # complementarity constraints
    # y[i+1] * GRFy[i] ≈ 0
    def contact_complementarity(m,n,f):
        if n < nfe:
            α = sum(m.foot_y[n+1,:,f])
            β = sum(m.GRFy[n,:,f])
            return α * β <= m.contact_penalty[n,f]
        else:
            return Constraint.Skip
    model.contact_complementarity = Constraint(model.fe, model.feet, rule=contact_complementarity)

    # # (μ * GRFy - ΣGRFx) * dx ≈ 0
    # # Only allows the foot to have horizontal velocity when it goes out of the friction cone
    # def friction_complementarity(m,n,f):
    #     α = m.friction_cone[n,f]
    #     β = sum(m.foot_dx[n,:,f,:])
    #     return α * β <= m.friction_penalty[n,f]
    # model.friction_complementarity = Constraint(model.fe, model.feet, rule=friction_complementarity)

    # # GRFx * dx ≈ 0
    # # ensures the friction force acts in the opposite direction to the foot velocity
    # def slip_complementarity(m,n,f,s):
    #     α = sum(m.GRFx[n,:,f,s])
    #     β = sum(m.foot_dx[n,:,f,s])
    #     return α * β <= m.slip_penalty[n,f,s]
    # model.slip_complementarity = Constraint(model.fe, model.feet, model.signs, rule=slip_complementarity)

    # GRFy * dx ≈ 0
    # ensures no foot slip
    def slip_complementarity(m,fe,f,s):
        α = sum(m.GRFy[fe,:,f]) # α = sum(m.GRFx[n,:,f,s])
        β = sum(m.foot_dx[fe,:,f,s])
        return α * β <= m.slip_penalty[fe,f,s]
    model.slip_complementarity = Constraint(model.fe, model.feet, model.signs, rule=slip_complementarity)

    #! JOINTS
    # dictionary to store joint links
    joints = ['front_knee', 'front_hip', 'back_hip', 'back_knee']
    model.J = Set(initialize=joints, ordered=True)
    motor_joints = ['front_hip', 'back_hip']
    model.J_m = Set(initialize=motor_joints, ordered=True)
    piston_joints = ['front_knee', 'back_knee']
    model.J_p = Set(initialize=piston_joints, ordered=True)

    model.F_c = Var(model.fe, model.cp, model.cDOF, model.J_p) # connection forces

    # connect piston end to attachment point
    def connect_piston_to_leg(m,n,cp,cdof,j):
        vars = m.q[n,cp,:]
        i = cDOF.index(cdof)
        if j == 'front_knee':
            return eom.piston_connection_front(*vars).flat[i] == 0
        elif j == 'back_knee':
            return eom.piston_connection_back(*vars).flat[i] == 0
        else:
            return Constraint.Skip
    model.connect_piston_to_leg = Constraint(model.fe, model.cp, model.cDOF, model.J_p, rule=connect_piston_to_leg)

    model.τ_m = Var(model.fe, model.J_m, bounds=(-motor.τ_max/sf, motor.τ_max/sf)) # actuator torque
    
    model.Fe_p = Var(model.fe, model.cp, model.J_p) # piston force
    model.Fr_p = Var(model.fe, model.cp, model.J_p) # piston force
    model.dFe_p = Var(model.fe, model.cp, model.J_p) # piston force derivative
    model.dFr_p = Var(model.fe, model.cp, model.J_p) # piston force derivative

    model.F_p = Var(model.fe, model.cp, model.J_p) # piston force
    # model.dF_p = Var(model.fe, model.cp, model.J_p) # piston force derivative
    model.F_r = Var(model.fe, model.cp, model.J_p, model.signs, bounds=(0, None)) # piston rebound force
    model.piston_mode = Set(initialize=['extend', 'retract'], ordered=True)
    model.u = Var(model.fe, model.J_p, model.piston_mode, bounds=(0, 1)) # piston valve command

    # joint limits (lower, upper)
    rom_limits = {'front_knee':(0, np.pi), 'front_hip':(-np.pi, 0), 'back_hip':(-np.pi, 0), 'back_knee':(0, np.pi)}
    model.joint_θ = Var(model.fe, model.cp, model.J, bounds=lambda m,n,cp,j: rom_limits[j])
    speed_limits = {'front_knee':(None, None), 'front_hip':(-motor.θ̇_max, motor.θ̇_max), 'back_hip':(-motor.θ̇_max, motor.θ̇_max), 'back_knee':(None, None)}
    model.joint_dθ = Var(model.fe, model.cp, model.J, bounds=lambda m,n,cp,j: speed_limits[j]) #rad/s

    # to hip : 0.05175 - 0.12009
    # to offset : 0.0480 - 0.1180
    # piston length limits
    for dof in ['l5', 'l6']:
        model.q[:,:,dof].setlb(0.048) # 0.048 # distance between piston rod and body CoM
        model.q[:,:,dof].setub(0.118) # 0.118
    # joint rebound torque complementarity
    model.rebound_penalty = Var(model.fe, model.J_p, model.signs, bounds=(0,1))
    # (θ - θ[i+1]) * τᵣ ≈ 0
    def piston_rebound_complementarity(m,n,j,s):
        if n < nfe:
            dof = {'front_knee': 'l5', 'back_knee': 'l6'}[j]
            if s == '+':
                α = sum(m.q[n+1,cp,dof].ub - m.q[n+1,cp,dof] for cp in m.cp)
                β = sum(m.F_r[n,:,j,'-'])
                return α * β <= m.rebound_penalty[n,j,s]
            else:
                α = sum(m.q[n+1,cp,dof] - m.q[n+1,cp,dof].lb for cp in m.cp)
                β = sum(m.F_r[n,:,j,'+'])
                return α * β <= m.rebound_penalty[n,j,s]
        else:
            return Constraint.Skip
    model.piston_rebound_complementarity = Constraint(model.fe, model.J_p, model.signs, rule=piston_rebound_complementarity)


    joint_links = {'front_knee':('θ1', 'θ2'), 'front_hip':('θ2', 'θ0'), 'back_hip':('θ0', 'θ3'), 'back_knee':('θ3', 'θ4')}
    def calc_joint_θ(m,n,cp,j): # CCW angle from parent to child
        θp, θc = joint_links[j]
        return m.joint_θ[n,cp,j] == np.pi - m.q[n,cp,θp] + m.q[n,cp,θc]
    model.calc_joint_θ = Constraint(model.fe, model.cp, model.J, rule=calc_joint_θ)

    def calc_joint_dθ(m,n,cp,j):
        θp, θc = joint_links[j]
        return m.joint_dθ[n,cp,j] == m.dq[n,cp,θc] - m.dq[n,cp,θp]
    model.calc_joint_dθ = Constraint(model.fe, model.cp, model.J, rule=calc_joint_dθ)

    #! ACTUATOR PROPERTIES
    # joint torque speed properties
    def motor_torque_speed_curve(m,fe,cp,j,dir):
        return motor.torque_speed_curve(sf*m.τ_m[fe,j], m.joint_dθ[fe,cp,j], dir)
    model.motor_torque_speed_curve = Constraint(model.fe, model.cp, model.J_m, model.signs, rule=motor_torque_speed_curve)
    # piston bang-bang
    model.piston_penalty = Var(model.fe, model.J_p, model.piston_mode, bounds=(0,1))
    def piston_complementarity(m,fe,j,mode):
        α = m.u[fe,j,mode]
        β = 1 - m.u[fe,j,mode]
        return α * β <= m.piston_penalty[fe,j,mode]
    model.piston_complementarity = Constraint(model.fe, model.J_p, model.piston_mode, rule=piston_complementarity)

    # # first order piston model
    # def piston_model(m,fe,cp,j):
    #     dof = {'front_knee': 'l5', 'back_knee': 'l6'}[j]
    #     return piston.model(m.u[fe,j,'extend'], m.u[fe,j,'retract'], sf*m.F_p[fe,cp,j], sf*m.dF_p[fe,cp,j], m.dq[fe,cp,dof])
    # model.piston_model = Constraint(model.fe, model.cp, model.J_p, rule=piston_model)

    # Fe = Fe_max - τ*dFe
    def extend_first_order_model(m,fe,cp,j):
        dof = {'front_knee': 'l5', 'back_knee': 'l6'}[j]
        return piston.extend_first_order_model(m.q[fe,cp,dof]-0.048, m.dq[fe,cp,dof], m.u[fe,j,'extend'], sf*m.Fe_p[fe,cp,j], sf*m.dFe_p[fe,cp,j])
    model.extend_first_order_model = Constraint(model.fe, model.cp, model.J_p, rule=extend_first_order_model)
    # Fr = Fr_max - τ*dFr
    def retract_first_order_model(m,fe,cp,j):
        dof = {'front_knee': 'l5', 'back_knee': 'l6'}[j]
        return piston.retract_first_order_model(m.q[fe,cp,dof]-0.048, -m.dq[fe,cp,dof], m.u[fe,j,'retract'], sf*m.Fr_p[fe,cp,j], sf*m.dFr_p[fe,cp,j])
    model.retract_first_order_model = Constraint(model.fe, model.cp, model.J_p, rule=retract_first_order_model)
    # F = Fe - Fr - c*dx
    def piston_force(m,fe,cp,j):
        dof = {'front_knee': 'l5', 'back_knee': 'l6'}[j]
        return piston.force(m.u[fe,j,'extend'], m.u[fe,j,'retract'], sf*m.F_p[fe,cp,j], sf*m.Fe_p[fe,cp,j], sf*m.Fr_p[fe,cp,j], m.dq[fe,cp,dof])
    model.piston_force = Constraint(model.fe, model.cp, model.J_p, rule=piston_force)

    # piston extend force implicit euler integration
    # model.piston_force_integration = Constraint(model.fe, model.cp, model.J_p, rule=piston.integration(model.F_p, model.dF_p))
    model.piston_extend_force_integration = Constraint(model.fe, model.cp, model.J_p, rule=piston.integration(model.Fe_p, model.dFe_p))
    model.piston_retract_force_integration = Constraint(model.fe, model.cp, model.J_p, rule=piston.integration(model.Fr_p, model.dFr_p))


    #! EQUATIONS OF MOTION
    def dynamics(m,fe,cp,gdof):
        j = gDOF.index(gdof)
        GRFx = [sf*(m.GRFx[fe,cp,'front_foot','+']-m.GRFx[fe,cp,'front_foot','-']), sf*(m.GRFx[fe,cp,'back_foot','+']-m.GRFx[fe,cp,'back_foot','-'])]
        GRFy = [sf*m.GRFy[fe,cp,f] for f in feet]
        τ_m = [sf*m.τ_m[fe,j] for j in m.J_m]
        F_p = [sf*m.F_p[fe,cp,j] + sf*(m.F_r[fe,cp,j,'+']-m.F_r[fe,cp,j,'-']) for j in m.J_p]
        F_c = [sf*m.F_c[fe,cp,'x','front_knee'], sf*m.F_c[fe,cp,'y','front_knee'], sf*m.F_c[fe,cp,'x','back_knee'], sf*m.F_c[fe,cp,'y','back_knee']]
        vars = [*m.q[fe,cp,:], *m.dq[fe,cp,:], *m.ddq[fe,cp,:], *τ_m, *F_p, *F_c, *GRFx, *GRFy]
        return eom.λ_EOM(*vars).flat[j] == 0

    model.dynamics = Constraint(model.fe, model.cp, model.gDOF, rule=dynamics)

    #! TYING UP LOOSE ENDS
    # These variables are left unconstrained
    model.GRFx[1,1,:,:].setub(1.0)
    model.GRFy[1,1,:].setub(1.0)
    model.GRFx[nfe,ncp,:,:].setub(1.0)
    model.GRFy[nfe,ncp,:].setub(1.0)
    # model.F_r[nfe,ncp,:,:].fix(0) # rebound force
    model.h[1].fix(1.0) # left out of integration

    return model



#! COST FUNCTIONS
# usually multiply this by 1e5
def penalty(m):
    return sum(m.contact_penalty[:,:]) + sum(m.slip_penalty[:,:,:]) + sum(m.rebound_penalty[:,:,:])

def piston_penalty_sum(m):
    return sum(m.piston_penalty[:,:,:])

def torque_squared(m):
    return sum(m.τ_m[fe,j]**2 for fe in m.fe for j in m.J_m)

def power_squared(m):
    return sum((m.τ_m[fe,j]*m.joint_dθ[fe,1,j])**2 for fe in m.fe for j in m.J_m)

def peak_power_squared(model):
    # add peak power calculation
    model.peak_motor_power = Var(model.J_m) # peak motor power
    def peak_motor_power_calc(m,fe,j):
        return (m.τ_m[fe,j]*m.joint_dθ[fe,j])**2 <= m.peak_motor_power[j]
    model.peak_motor_power_calc = Constraint(model.fe, model.J_m, rule=peak_motor_power_calc)
    return sum(model.peak_motor_power[:])

def time_sum(m):
    return sum(m.h[:])

def pose_error(m, q_des):
    pose_error_squared = sum([(m.q[fe,cp,dof] - q_des[fe,cp,dof])**2 for fe in m.fe for cp in m.cp for dof in m.gDOF])
    return pose_error_squared

# stops limbs swinging wildly by trying to maintain final pose throughout motion (excluding body x,y,θ)
def final_pose_error_squared(m):
    nfe = len(m.fe)
    return sum([(m.q[fe,cp,dof] - m.q[nfe,cp,dof])**2 for fe in m.fe for cp in m.cp for dof in m.gDOF if dof not in ['x0','y0','θ0']])

# CoT objective
def COT(m):
    nfe, ncp = len(m.fe), len(m.cp)
    v̅ = (m.q[nfe,ncp,'x0'] - m.q[1,ncp,'x0']) / sum(m.h[:]) # average velocity
    return power_squared(m) / eom.BW * v̅
# Power input / mgv

def foot_height_sum(m):
    return sum(m.foot_y[:,:,:])

# piston actuated movement
def piston_actuated_movement(m):
    front = sum([(1-m.u[fe,'front_knee','extend']-m.u[fe,'front_knee','retract']+m.u[fe,'front_knee','extend']*m.u[fe,'front_knee','extend'])*(m.dq[fe,1,'l5']**2) for fe in m.fe])
    back = sum([(1-m.u[fe,'back_knee','extend']-m.u[fe,'back_knee','retract']+m.u[fe,'back_knee','extend']*m.u[fe,'back_knee','extend'])*(m.dq[fe,1,'l6']**2) for fe in m.fe])
    return front + back

# minimise the amount of time the piston spends unactuated
def piston_unactuated(m):
    return sum([(1-m.u[fe,j,'extend']-m.u[fe,j,'retract']+m.u[fe,j,'extend']*m.u[fe,j,'extend']) for fe in m.fe for j in m.J_p])


def piston_length_sum(m):
    return sum(m.q[:,:,'l5']) + sum(m.q[:,:,'l6'])