import cloudpickle
import numpy as np
import pandas as pd
from kemba.dynamics import BW
import sys
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from visualise import get_time_arrays

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# padding trajectory on either side with constant values
start_delay = 10 # seconds
end_delay = 10 # seconds

# loading saved model
with open('./kemba/cache/jump-land-0.7.pkl', mode='rb') as file:
    model = cloudpickle.load(file)

nfe, ncp = len(model.fe), len(model.cp)
time, time_cp = get_time_arrays(model)
time += start_delay
ctrl_dt = 5e-4
ctrl_time = np.arange(0, time[-1]+end_delay, ctrl_dt)

piston_dead_time = 6e-3 # s

# get data from model
# front motor command
θm_f = [-model.joint_θ[fe,ncp,'front_hip'].value-np.pi for fe in model.fe]
θ̇m_f = [-model.joint_dθ[fe,ncp,'front_hip'].value for fe in model.fe]
τm_f = [BW*model.τ_m[fe,'front_hip'].value for fe in model.fe]
# back motor command
θm_b = [model.joint_θ[fe,ncp,'back_hip'].value for fe in model.fe]
θ̇m_b = [model.joint_dθ[fe,ncp,'back_hip'].value for fe in model.fe]
τm_b = [BW*model.τ_m[fe,'back_hip'].value for fe in model.fe]
# front piston command (round to nearest integer [0,1])
ue_f = np.rint([model.u[fe,'front_knee','extend'].value for fe in model.fe])
ur_f = np.rint([model.u[fe,'front_knee','retract'].value for fe in model.fe])
# front piston desired state
xp_f = [model.q[fe,ncp,'l5'].value-0.048 for fe in model.fe]
ẋp_f = [model.dq[fe,ncp,'l5'].value for fe in model.fe]
Fp_f = [BW*model.F_p[fe,ncp,'front_knee'].value for fe in model.fe]
# back piston command (round to nearest integer [0,1])
ue_b = np.rint([model.u[fe,'back_knee','extend'].value for fe in model.fe])
ur_b = np.rint([model.u[fe,'back_knee','retract'].value for fe in model.fe])
# back piston desired state
xp_b = [model.q[fe,ncp,'l6'].value-0.048 for fe in model.fe]
ẋp_b = [model.dq[fe,ncp,'l6'].value for fe in model.fe]
Fp_b = [BW*model.F_p[fe,ncp,'back_knee'].value for fe in model.fe]

# body state
x = [model.q[fe,ncp,'x0'].value for fe in model.fe]
y = [model.q[fe,ncp,'y0'].value for fe in model.fe]
r = [model.q[fe,ncp,'θ0'].value for fe in model.fe]
dx = [model.dq[fe,ncp,'x0'].value for fe in model.fe]
dy = [model.dq[fe,ncp,'y0'].value for fe in model.fe]
dr = [model.dq[fe,ncp,'θ0'].value for fe in model.fe]
ddx = [model.ddq[fe,ncp,'x0'].value for fe in model.fe]
ddy = [model.ddq[fe,ncp,'y0'].value for fe in model.fe]

# interpolate data
# front motor
θm_f = interp1d(time, θm_f, kind='quadratic', fill_value=(θm_f[0], θm_f[-1]), bounds_error=False)
θ̇m_f = interp1d(time, θ̇m_f, kind='linear', fill_value=0, bounds_error=False)
τm_f = interp1d(time, τm_f, kind='linear', fill_value=0, bounds_error=False)
# back motor
θm_b = interp1d(time, θm_b, kind='quadratic', fill_value=(θm_b[0], θm_b[-1]), bounds_error=False)
θ̇m_b = interp1d(time, θ̇m_b, kind='linear', fill_value=0, bounds_error=False)
τm_b = interp1d(time, τm_b, kind='linear', fill_value=0, bounds_error=False)
# front piston
ue_f = interp1d(time-piston_dead_time, ue_f, kind='zero', fill_value=0, bounds_error=False)
ur_f = interp1d(time-piston_dead_time, ur_f, kind='zero', fill_value=0, bounds_error=False)
xp_f = interp1d(time, xp_f, kind='quadratic', fill_value=0, bounds_error=False)
ẋp_f = interp1d(time, ẋp_f, kind='linear', fill_value=0, bounds_error=False)
Fp_f = interp1d(time, Fp_f, kind='quadratic', fill_value=0, bounds_error=False)
# back piston
ue_b = interp1d(time-piston_dead_time, ue_b, kind='zero', fill_value=0, bounds_error=False)
ur_b = interp1d(time-piston_dead_time, ur_b, kind='zero', fill_value=0, bounds_error=False)
xp_b = interp1d(time, xp_b, kind='quadratic', fill_value=0, bounds_error=False)
ẋp_b = interp1d(time, ẋp_b, kind='linear', fill_value=0, bounds_error=False)
Fp_b = interp1d(time, Fp_b, kind='quadratic', fill_value=0, bounds_error=False)
# body state
x = interp1d(time, x, kind='quadratic', fill_value=(x[0], x[-1]), bounds_error=False)
y = interp1d(time, y, kind='quadratic', fill_value=(y[0], y[-1]), bounds_error=False)
r = interp1d(time, r, kind='quadratic', fill_value=(r[0], r[-1]), bounds_error=False)
dx = interp1d(time, dx, kind='linear', fill_value=(0, 0), bounds_error=False)
dy = interp1d(time, dy, kind='linear', fill_value=(0, 0), bounds_error=False)
dr = interp1d(time, dr, kind='linear', fill_value=(0, 0), bounds_error=False)
ddx = interp1d(time, ddx, kind='zero', fill_value=(0, 0), bounds_error=False)
ddy = interp1d(time, ddy, kind='zero', fill_value=(0, 0), bounds_error=False)


# assign to dataframe
trajs = pd.DataFrame({'time':ctrl_time})
trajs['qm_f'] = θm_f(ctrl_time)
trajs['wm_f'] = θ̇m_f(ctrl_time)
trajs['tm_f'] = τm_f(ctrl_time)

trajs['qm_b'] = θm_b(ctrl_time)
trajs['wm_b'] = θ̇m_b(ctrl_time)
trajs['tm_b'] = τm_b(ctrl_time)

trajs['ue_f'] = ue_f(ctrl_time)
trajs['ur_f'] = ur_f(ctrl_time)
trajs['xp_f'] = xp_f(ctrl_time)
trajs['vp_f'] = ẋp_f(ctrl_time)
trajs['Fp_f'] = Fp_f(ctrl_time)

trajs['ue_b'] = ue_b(ctrl_time)
trajs['ur_b'] = ur_b(ctrl_time)
trajs['xp_b'] = xp_b(ctrl_time)
trajs['vp_b'] = ẋp_b(ctrl_time)
trajs['Fp_b'] = Fp_b(ctrl_time)

trajs['x'] = x(ctrl_time)
trajs['y'] = y(ctrl_time)
trajs['r'] = r(ctrl_time)
trajs['dx'] = dx(ctrl_time)
trajs['dy'] = dy(ctrl_time)
trajs['dr'] = dr(ctrl_time)
trajs['ddx'] = ddx(ctrl_time)
trajs['ddy'] = ddy(ctrl_time)

fig, ax = plt.subplots()
ax.scatter(time, ue_f(time))
ax.plot(ctrl_time, ue_f(ctrl_time), color='r')

trajs.to_csv('traj.csv', sep=',', encoding='utf-8', index=False)

plt.show()
