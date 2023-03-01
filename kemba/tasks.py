from pyomo.environ import*
from math import pi


# TODO

def rest_pose(m,fe):
    m.dq[fe,:,:].fix(0) # stationary
    m.foot_y[fe,:,:].fix(0) # feet on the ground
    m.q[fe,:,'l5'].fix(m.q[fe,:,'l5'].lb) # piston retracted
    m.q[fe,:,'l6'].fix(m.q[fe,:,'l6'].lb) # piston retracted


def periodic_bounding(model, stride):
    '''
    t = 0.5s
    nfe = 60
    stride <= 1.0
    timings = {'front_foot':(0.2, 0.4), 'back_foot':(0.6, 0.8)}
    '''
    nfe, ncp = len(model.fe), len(model.cp)
    mid = int(0.5*nfe)
    # reasonable body height range
    model.q[:,:,'y0'].setub(0.45)
    model.q[:,:,'y0'].setlb(0.2)
    # reasonable body angle range
    model.q[:,:,'θ0'].setub(0.3)
    model.q[:,:,'θ0'].setlb(-0.3)
    # reasonable joint angle ranges
    model.joint_θ[:,:,'front_hip'].setlb(1.6*pi)
    model.joint_θ[:,:,'back_hip'].setlb(1.6*pi)
    model.joint_θ[:,:,'front_hip'].setub(1.9*pi)
    model.joint_θ[:,:,'back_hip'].setub(1.9*pi)
    model.joint_θ[:,:,'front_knee'].setub(0.45*pi)
    model.joint_θ[:,:,'back_knee'].setub(0.45*pi)
    # stop pistons bottoming out
    # model.q[:,:,'l5'].setlb(0.049) # min 0.048
    # model.q[:,:,'l6'].setlb(0.049) # min 0.048

    # periodic position
    def ss_p(m,dof):
        if dof == 'x0':
            return Constraint.Skip
        else:
            return m.q[1,ncp,dof] == m.q[nfe,ncp,dof]
    model.ss_p = Constraint(model.gDOF, rule=ss_p)
    # periodic velocity
    def ss_v(m,dof):
        return m.dq[1,ncp,dof] == m.dq[nfe,ncp,dof]
    model.ss_v = Constraint(model.gDOF, rule=ss_v)

    # initial condition - extended flight phase
    model.q[1,ncp,'θ0'].fix(0)
    model.q[1,ncp,'x0'].fix(0)
    # middle condition - collected suspension
    model.q[mid,ncp,'θ0'].fix(0)
    # final condition
    model.q[nfe,ncp,'x0'].setlb(stride)

    # # no back foot contact for first half
    # for fe in range(1, mid):
    #     model.GRFy[fe,:,'back_foot'].fix(0.0)
    # # no front foot contact for second half
    # for fe in range(mid, nfe+1):
    #     model.GRFy[fe,:,'front_foot'].fix(0.0)


def prescribe_contact_order(m, timings, min_foot_height=5e-3, min_GRFy=1e-3):
    '''
    timings = {'front_foot':(0.2, 0.4), 'back_foot':(0.6, 0.8)}
    '''
    nfe, ncp = len(m.fe), len(m.cp)
    def inclusive_range(start, stop): return range(start, stop+1)
    def fix(foot, start, stop):
        # phase 1: flight
        for fe in inclusive_range(1, start-1):
            for cp in m.cp:
                m.foot_y[fe,cp,foot].setlb(min_foot_height)
                m.GRFy[fe,cp,foot].fix(0)
        # phase 2: stance
        for fe in inclusive_range(start+1, stop-1):
            for cp in m.cp:
                m.foot_y[fe,cp,foot].fix(0)
                m.GRFy[fe,cp,foot].setlb(min_GRFy)
        # phase 3: flight
        for fe in inclusive_range(stop+1, nfe):
            for cp in m.cp:
                m.foot_y[fe,cp,foot].setlb(min_foot_height)
                m.GRFy[fe,cp,foot].fix(0)
    
    for foot, (start, stop) in timings.items():
        fe_start, fe_stop = int(nfe * start), int(nfe * stop)
        fix(foot, fe_start, fe_stop)


def jump(m, height):
    '''
    t = 0.55s
    nfe = 100
    Boom max jump height about 1.7m
    Works well with t=0.55s and nfe=100
    '''
    nfe, ncp = len(m.fe), len(m.cp)
    # initial conditions
    m.q[:,ncp,'x0'].fix(0)
    m.q[:,ncp,'θ0'].fix(0) # level throughout
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    # m.joint_θ[1,ncp,'front_hip'].fix(-0.32)
    # m.joint_θ[1,ncp,'back_hip'].fix(-0.32)
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    # final conditions
    # m.joint_θ[nfe,ncp,'front_hip'].fix(-1.0)
    # m.joint_θ[nfe,ncp,'back_hip'].fix(-1.0)
    m.q[nfe,ncp,'l5'].fix(m.q[1,ncp,'l5'].ub) # piston extended
    m.q[nfe,ncp,'l6'].fix(m.q[1,ncp,'l6'].ub) # piston extended
    # apex condition
    # m.dq[nfe,ncp,'y0'].fix(0)
    # m.dq[nfe,ncp,'θ0'].fix(0)
    # m.dq[nfe,ncp,'θ0'].fix(0)
    m.q[nfe,ncp,'y0'].setlb(height)


def jump_forwards(m, distance):
    nfe, ncp = len(m.fe), len(m.cp)
    # initial conditions
    m.q[1,ncp,'x0'].fix(0)
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    # final conditions
    m.GRFy[nfe,ncp,:].fix(0) # feet off the ground
    m.foot_y[nfe,ncp,:].setlb(0.05) # feet off the ground
    m.q[nfe,ncp,'x0'].setlb(distance)


def jump_and_land(m, height):
    '''
    t = 0.8s
    nfe = 80
    timestep_bounds=(0.8,1.2)
    Note: Doesn't like using the pistons for landing
    '''
    nfe, ncp = len(m.fe), len(m.cp)
    mid = int(nfe/2)
    # initial conditions
    m.q[:,ncp,'x0'].fix(0) # x = 0 throughout
    m.q[:,ncp,'θ0'].fix(0) # body level throughout
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    # apex condition
    m.q[mid,ncp,'y0'].setlb(height)
    # m.foot_y[int(nfe/2),ncp,:].setlb(0.01) # feet at least 10cm above ground
    # final conditions
    # included small piston offset so that it does not rely on the rebound force
    m.q[nfe-4,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb + 1e-2) # piston retracted
    m.q[nfe-4,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb + 1e-2) # piston retracted
    m.foot_y[nfe,ncp,:].fix(0.0) # feet on the ground
    m.dq[nfe,ncp,:].fix(0) # stationary


def backflip(m):
    '''
    t = 1.0s
    nfe = 100
    timestep_bounds=(0.8,1.2)
    timings = {'front_foot':(0.2, 0.4), 'back_foot':(0.6, 0.8)}
    Note: Struggles to land because of the angle bounds on the hip joints
    '''
    nfe, ncp = len(m.fe)-1, len(m.cp)
    # initial conditions
    m.q[1,ncp,'x0'].fix(0)
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    # apex condition
    apex = int(0.7*nfe)
    m.q[apex,ncp,'y0'].setlb(0.7) # must be in the air
    # m.dq[nfe,ncp,'θ0'].setlb(6.0) # body rotation rate
    m.q[apex,ncp,'θ0'].setub(-pi) # body upside down
    # m.q[apex,ncp,'l5'].setub(m.q[apex,ncp,'l5'].lb+1e-2) # piston retracted
    # m.q[apex,ncp,'l6'].setub(m.q[apex,ncp,'l6'].lb+1e-2) # piston retracted
    # m.q[int(nfe/2),:,'θ0'].fix(pi)
    # m.foot_y[nfe,:,:].setlb(0.01)
    # final conditions
    # m.foot_y[nfe,ncp,:].fix(0) # feet on the ground
    m.dq[nfe,ncp,:].fix(0) # stationary
    m.q[nfe,ncp,'θ0'].setub(-1.9*pi)
    m.q[nfe,ncp,'x0'].setub(1.0)
    m.q[nfe,ncp,'x0'].setlb(-1.0)
    m.q[nfe,ncp,'l5'].setub(m.q[nfe,ncp,'l5'].lb+2e-2) # piston retracted
    m.q[nfe,ncp,'l6'].setub(m.q[nfe,ncp,'l6'].lb+2e-2) # piston retracted


def initiate_bounding(m, speed):
    '''
    Accelerate from rest into bounding controller
    t = 0.4s
    nfe = 50
    '''
    nfe, ncp = len(m.fe), len(m.cp)
    #* initial condition
    m.q[1,ncp,'x0'].fix(0)
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # pistons retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # pistons retracted
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    #* apex condition - bound apex
    apex = int(nfe - 1)
    # m.q[apex,ncp,'x0'].setlb(0.2) # minimum forward distance
    m.q[apex,ncp,'y0'].fix(0.30) # body height
    m.q[apex,ncp,'θ0'].fix(0) # body level
    m.dq[apex,ncp,'x0'].fix(speed) # forward speed
    # m.dq[nfe,ncp,'y0'].fix(0)
    m.dq[apex,ncp,'θ0'].setub(-3.0) # body pitch rate
    # m.joint_θ[apex,ncp,'front_hip'].fix(-0.3)
    # m.joint_θ[apex,ncp,'back_hip'].fix(-0.4)
    # m.foot_y[nfe,ncp,:].setlb(1e-2) # feet off ground
    m.q[apex,ncp,'l5'].setub(m.q[nfe,ncp,'l5'].lb+10e-3) # pistons retracted
    m.q[apex,ncp,'l6'].setub(m.q[nfe,ncp,'l6'].lb+10e-3) # pistons retracted
    #* final condition
    # m.dq[nfe,ncp,'x0'].fix(speed) # forward speed
    # m.dq[nfe,ncp,'θ0'].setub(-2.0) # body pitch rate
    # m.q[nfe,ncp,'l5'].fix(m.q[nfe,ncp,'l5'].lb) # pistons retracted
    # m.q[nfe,ncp,'l6'].fix(m.q[nfe,ncp,'l6'].lb) # pistons retracted


def terminate_bounding(m):
    nfe, ncp = len(m.fe), len(m.cp)
    # initial conditions
    m.q[1,ncp,'x0'].fix(0)
    m.q[1,ncp,'y0'].fix(0.29)
    m.q[1,ncp,'θ0'].fix(0) # body level
    m.dq[1,ncp,'x0'].fix(1.75) # forward speed
    m.dq[2,ncp,'x0'].fix(1.75) # forward speed
    m.dq[1,ncp,'y0'].fix(0)
    m.dq[1,ncp,'θ0'].fix(-3.0)
    m.q[1,ncp,'l5'].fix(m.q[nfe,ncp,'l5'].lb) # pistons retracted
    m.q[1,ncp,'l6'].fix(m.q[nfe,ncp,'l6'].lb) # pistons retracted
    m.joint_θ[1,ncp,'front_hip'].fix(-0.8) # right hip position
    m.joint_θ[1,ncp,'back_hip'].fix(-0.1) # left hip position
    # final condition
    m.q[nfe,ncp,'l5'].fix(m.q[nfe,ncp,'l5'].lb) # pistons retracted
    m.q[nfe,ncp,'l6'].fix(m.q[nfe,ncp,'l6'].lb) # pistons retracted
    m.dq[nfe,ncp,'x0'].fix(0) # zero forward speed
    m.foot_y[nfe,ncp,:].fix(0.0) # feet on the ground


def popup(m, height=0.29):
    '''
    Jump into bounding controller from rest
    t = 0.3s
    nfe = 60
    '''
    nfe, ncp = len(m.fe), len(m.cp)
    # initial condition
    m.q[1,ncp,'x0'].fix(0)
    m.q[1,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[1,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    m.dq[1,ncp,:].fix(0) # stationary
    m.foot_y[1,ncp,:].fix(0.0) # feet on the ground
    # final condition
    m.q[nfe,ncp,'y0'].fix(height)
    m.q[nfe,ncp,'θ0'].fix(0)
    m.dq[nfe,ncp,'x0'].setub(0.2)
    m.dq[nfe,ncp,'x0'].setlb(-0.2)
    m.dq[nfe,ncp,'y0'].fix(0)
    m.dq[nfe,ncp,'θ0'].fix(3.5)
    m.joint_θ[nfe,ncp,'front_hip'].fix(-0.45)
    m.joint_θ[nfe,ncp,'back_hip'].fix(-0.45)
    m.q[nfe-1,ncp,'l5'].fix(m.q[nfe,ncp,'l5'].lb) # piston retracted
    m.q[nfe-1,ncp,'l6'].fix(m.q[nfe,ncp,'l6'].lb) # piston retracted


#! Sanity Checks
def drop_test(model, height):
    nfe, ncp = len(model.fe), len(model.cp)
    start_pose = {'front_knee':0.5*pi, 'front_hip':-0.1*pi, 'back_hip':-0.2*pi, 'back_knee':0.5*pi}
    end_pose = {'front_knee':0.5*pi, 'front_hip':-0.1*pi, 'back_hip':-0.2*pi, 'back_knee':0.5*pi}
    model.q[1,ncp,'x0'].fix(0.0)
    model.q[1,ncp,'y0'].fix(height)
    model.q[1,ncp,'θ0'].fix(0.0)
    # set to rest position
    model.dq[1,ncp,:].fix(0.0) # stationary
    # set initial pose
    for j in model.J:
        model.joint_θ[1,ncp,j].fix(start_pose[j])
    # final pose
    for j in model.J:
        model.joint_θ[nfe,ncp,j].fix(end_pose[j])
    # zero actuators
    # model.τ_m[:,:].fix(0.0)
    # model.u[:,:,:].fix(0.0)
    # model.F_p[:,:,:].fix(0.0)


def stand(m):
    nfe, ncp = len(m.fe), len(m.cp)
    # initial condition
    m.q[1,ncp,'y0'].fix(0.21)
    m.q[1,ncp,'x0'].fix(0.0)
    m.q[1,ncp,'θ0'].fix(0.0)
    m.dq[1,ncp,:].fix(0) # stationary
    m.q[:,ncp,'l5'].fix(m.q[1,ncp,'l5'].lb) # piston retracted
    m.q[:,ncp,'l6'].fix(m.q[1,ncp,'l6'].lb) # piston retracted
    m.F_p[:,ncp,:].fix(0)
    # m.foot_y[:,ncp,:].fix(0.0) # feet on the ground
    m.joint_θ[nfe,ncp,'front_hip'].fix(-0.1*pi)
    m.joint_θ[nfe,ncp,'back_hip'].fix(-0.1*pi)
    # final condition
    # m.dq[:,:,:].fix(0) # stationary throughout
