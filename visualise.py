import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from pyomo.environ import value
from scipy.interpolate import CubicHermiteSpline
# from kemba.dynamics import BW
from kemba.dynamics import BW, x, y, λ_points

savefig = False
description = 'accelerate'

if savefig: matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def stop_motion(model, n_frames):
    fig, ax = plt.subplots()
    fig.set_figwidth(6)
    ax.set_aspect('equal')
    ax.grid(True, which='major', axis='both', linewidth=0.25)

    nfe, ncp = len(model.fe), len(model.cp)

    ys = [model.q[fe,ncp,'y0'].value for fe in model.fe]
    y_min, y_max = np.min(ys[1:-2]), np.max(ys[1:-2])
    ax.set_ylim([0, 0.5+y_max])

    def draw_frame(fe,m,ax,x0,alpha=1.0):
        # plotting piston
        vars = [m.q[fe,ncp,dof].value for dof in m.gDOF]
        points = λ_points(*vars)
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = points[:,:]

        front_piston_mode = round(model.u[fe,'front_knee','extend'].value) - round(model.u[fe,'front_knee','retract'].value) + 1
        # front_piston_mode = 1
        back_piston_mode = round(model.u[fe,'back_knee','extend'].value) - round(model.u[fe,'back_knee','retract'].value) + 1
        # back_piston_mode = 1
        piston_colors = ['green', 'orange', 'red'] # retract, unactuated/both actuated, extend
        ax.plot([p9[x]+x0, p7[x]+x0], [p9[y], p7[y]], color=piston_colors[front_piston_mode], linewidth=1.5, alpha=alpha) # front piston
        ax.plot([p3[x]+x0, p9[x]+x0], [p3[y], p9[y]], color='k', linewidth=1.5, solid_capstyle='round', alpha=alpha)
        ax.plot([p10[x]+x0, p8[x]+x0], [p10[y], p8[y]], color=piston_colors[back_piston_mode], linewidth=1.5, alpha=alpha) # back piston
        ax.plot([p4[x]+x0, p10[x]+x0], [p4[y], p10[y]], color='k', linewidth=1.5, solid_capstyle='round', alpha=alpha)
        # plot body
        xs, ys = points[:6,0], points[:6,1]
        xs += x0
        ax.plot(xs, ys, color='k', linewidth=1.5, solid_capstyle='round', alpha=alpha)
        # plot GRF vectors
        GRFf = [model.GRFx[fe,ncp,'front_foot','+'].value - model.GRFx[fe,ncp,'front_foot','-'].value, model.GRFy[fe,ncp,'front_foot'].value]
        GRFb = [model.GRFx[fe,ncp,'back_foot','+'].value - model.GRFx[fe,ncp,'back_foot','-'].value, model.GRFy[fe,ncp,'back_foot'].value]
        ax.quiver([p1[x], p6[x]], [p1[y], p6[y]], [GRFf[x], GRFb[x]], [GRFf[y], GRFb[y]], scale=0.05*BW, units='xy', angles='xy', color='r', alpha=alpha, width=0.02)

    frames = np.linspace(1, nfe-1, n_frames, dtype=int)
    times, _ = get_time_arrays(model)
    times = np.take(times, frames-1)
    x0s = []
    for fe in frames:
        spacing = 0.6 # m
        # zero body x position and add spacing
        x0 = -model.q[fe,ncp,'x0'].value + ((n_frames*spacing)/nfe)*fe
        x0s.append(((n_frames*spacing)/nfe)*fe)
        draw_frame(fe, model, ax, x0)
    times = ["%.2f" % t for t in times]
    ax.xaxis.set_ticks(x0s)
    ax.xaxis.set_ticklabels(times)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    fig.tight_layout()
    if savefig: fig.savefig(f'figures/{description}-frames-plot.pdf', bbox_inches='tight')


def animate(model, speed=1.0, follow=False, GIF=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    nfe, ncp = len(model.fe), len(model.cp)
    hm = model.hm.value
    gDOF = model.gDOF
    # x,y = dynamics.x, dynamics.y

    model.h[1].value = 1.0 # can be left as NoneType

    # calculate frames for variable timestep
    h = np.array([model.h[fe].value for fe in model.fe])
    h_min = min(h)
    dur = np.rint(h / h_min).astype(int)
    frames = []
    for i, h in enumerate(dur):
        frames.extend([i+1] * h)
    xs = [model.q[fe,ncp,'x0'].value - (model.q[fe,ncp,'x0'].value if follow else 0) for fe in model.fe]
    ys = [model.q[fe,ncp,'y0'].value for fe in model.fe]
    x_min, x_max = np.min(xs[1:-2]), np.max(xs[1:-2])
    y_min, y_max = np.min(ys[1:-2]), np.max(ys[1:-2])

    # update function for animation
    def plot(fe,m,ax):
        ax.clear()
        ax.set_xlim([x_min-0.6, x_max+0.6])
        ax.set_ylim([0, y_max+0.5])
        if follow: ax.set_xticks([])
        # plotting piston
        vars = [m.q[fe,ncp,dof].value for dof in gDOF]
        points = λ_points(*vars)
        x0 = m.q[fe,ncp,'x0'].value if follow else 0
        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = points[:,:]

        front_piston_mode = round(model.u[fe,'front_knee','extend'].value) - round(model.u[fe,'front_knee','retract'].value) + 1
        # front_piston_mode = 1
        back_piston_mode = round(model.u[fe,'back_knee','extend'].value) - round(model.u[fe,'back_knee','retract'].value) + 1
        # back_piston_mode = 1
        piston_colors = ['green', 'orange', 'red'] # retract, unactuated/both actuated, extend

        ax.plot([p9[x]-x0, p7[x]-x0], [p9[y], p7[y]], color=piston_colors[front_piston_mode], linewidth=2) # front piston
        ax.plot([p3[x]-x0, p9[x]-x0], [p3[y], p9[y]], color='k', linewidth=3, solid_capstyle='round')
        ax.plot([p10[x]-x0, p8[x]-x0], [p10[y], p8[y]], color=piston_colors[back_piston_mode], linewidth=2) # back piston
        ax.plot([p4[x]-x0, p10[x]-x0], [p4[y], p10[y]], color='k', linewidth=3, solid_capstyle='round')
        # plotting body
        xs, ys = points[:6,0], points[:6,1]
        xs -= x0
        ax.plot(xs, ys, color='k', linewidth=3, solid_capstyle='round')
        # plotting GRF vectors
        GRFf = [model.GRFx[fe,ncp,'front_foot','+'].value - model.GRFx[fe,ncp,'front_foot','-'].value, model.GRFy[fe,ncp,'front_foot'].value]
        GRFb = [model.GRFx[fe,ncp,'back_foot','+'].value - model.GRFx[fe,ncp,'back_foot','-'].value, model.GRFy[fe,ncp,'back_foot'].value]
        ax.quiver([p1[x], p6[x]], [p1[y], p6[y]], [GRFf[x], GRFb[x]], [GRFf[y], GRFb[y]], scale=0.5*BW, units='xy', angles='xy', color='r')

    update = lambda fe: plot(fe,model,ax)

    dt_min = hm*h_min
    animation = FuncAnimation(fig, update, frames, interval=dt_min*1000, repeat=True, blit=False)
    writer = ImageMagickWriter() if GIF else FFMpegWriter(fps=speed*(1.0/dt_min), bitrate=5000) # metadata=dict(artist='Christopher Mailer')
    # ext = '.gif' if gif else '.mp4'
    animation.save(f'./kemba/{model.name}.{"gif" if GIF else "mp4"}', writer=writer)


def print_stats(m):
    hm = m.hm.value
    time_sum = hm*sum(m.h[fe].value for fe in m.fe)
    print(f'Duration: {time_sum:.3f}s')
    print(f'nfe: {len(m.fe)}')
    contact_penalty = sum([m.contact_penalty[fe,foot].value for fe in m.fe for foot in m.feet])
    print(f'Contact Penalty: {contact_penalty:.3e}')
    print(f'Objective function: {value(m.cost):.3e}')


def get_time_arrays(m):
    timesteps = [m.h[fe].value * m.hm for fe in m.fe]
    times = np.cumsum(timesteps)
    times = np.delete(np.insert(times, 0, 0), -1)
    nfe, ncp = len(m.fe), len(m.cp)
    times_with_cp = np.interp(np.arange(nfe*ncp-(ncp-1)), np.arange(nfe)*ncp, times) # linear interpolate
    for _ in range(ncp-1): # can't interpolate end collocation points so just appending hm/ncp
        times_with_cp = np.append(times_with_cp, times_with_cp[-1] + m.hm/ncp)
    return times, times_with_cp


def plot_penalties(model):
    fig, ax = plt.subplots(4, sharex=True)
    time, time_cp = get_time_arrays(model)
    ax[0].set_title('Contact penalty')
    ax[0].plot(time, [model.contact_penalty[fe,'front_foot'].value for fe in model.fe])
    ax[0].plot(time, [model.contact_penalty[fe,'back_foot'].value for fe in model.fe])
    ax[1].set_title('Friction penalty')
    ax[1].plot(time, [model.friction_penalty[fe,'front_foot'].value for fe in model.fe])
    ax[1].plot(time, [model.friction_penalty[fe,'back_foot'].value for fe in model.fe])
    ax[2].set_title('Slip penalty')
    ax[2].plot(time, [model.slip_penalty[fe,'front_foot', '+'].value - model.slip_penalty[fe,'front_foot', '-'].value for fe in model.fe])
    ax[2].plot(time, [model.slip_penalty[fe,'back_foot', '+'].value - model.slip_penalty[fe,'back_foot', '-'].value for fe in model.fe])
    ax[3].set_title('Rebound penalty')
    ax[3].plot(time, [model.rebound_penalty[fe,'front_knee', '+'].value - model.rebound_penalty[fe,'front_knee', '-'].value for fe in model.fe])
    ax[3].plot(time, [model.rebound_penalty[fe,'back_knee', '+'].value - model.rebound_penalty[fe,'back_knee', '-'].value for fe in model.fe])
    fig.tight_layout()


def plot_actuators(model):
    fig, ax = plt.subplots(3, sharex=True)
    fig.set_figwidth(6)
    fig.set_figheight(4)
    time, time_cp = get_time_arrays(model)
    nfe, ncp = len(model.fe), len(model.cp)
    # torques
    ax[0].set_title('Hip Torque')
    ax[0].grid(True, which='major', axis='both', linewidth=0.5)
    ax[0].plot(time, [BW*model.τ_m[fe,'front_hip'].value for fe in model.fe], label='Right')
    ax[0].set_ylabel('Torque (Nm)')
    ax[0].plot(time, [BW*model.τ_m[fe,'back_hip'].value for fe in model.fe], label='Left')
    ax[0].legend(loc='upper left')

    ax[1].set_title('Right Knee Piston')
    ax[1].grid(True, which='major', axis='both', linewidth=0.5)
    ax[1].step(time, [model.u[fe,'front_knee','extend'].value for fe in model.fe], label='Extend', where='post')
    ax[1].step(time, [model.u[fe,'front_knee','retract'].value for fe in model.fe], label='Retract', where='post')
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels(['Off', 'On'])
    ax[1].set_ylabel('Valve')
    ax2 = ax[1].twinx()
    ax2.grid(True, which='major', axis='both', linewidth=0.5)
    # F_p_front = CubicHermiteSpline(time, [BW*model.F_p[fe,ncp,'front_knee'].value for fe in model.fe], [BW*model.dF_p[fe,ncp,'front_knee'].value for fe in model.fe])
    # time_new = np.linspace(time[0], time[-1], num=1000, endpoint=True)
    # ax2.plot(time_new, F_p_front(time_new), c='tab:red')
    ax2.plot(time, [BW*model.F_p[fe,ncp,'front_knee'].value for fe in model.fe], c='tab:red')
    ax2.set_ylabel('Force (N)')

    ax[1].legend(loc='best')
    ax[2].set_title('Left Knee Piston')
    ax[2].grid(True, which='major', axis='both', linewidth=0.5)
    ax[2].step(time, [model.u[fe,'back_knee','extend'].value for fe in model.fe], label='Extend', where='post')
    ax[2].step(time, [model.u[fe,'back_knee','retract'].value for fe in model.fe], label='Retract', where='post')
    ax[2].set_yticks([0,1])
    ax[2].set_yticklabels(['Off', 'On'])
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Valve')
    ax[2].legend(loc='best')
    ax3 = ax[2].twinx()
    ax3.grid(True, which='major', axis='both', linewidth=0.5)
    # F_p_back = CubicHermiteSpline(time, [BW*model.F_p[fe,ncp,'back_knee'].value for fe in model.fe], [BW*model.dF_p[fe,ncp,'back_knee'].value for fe in model.fe])
    # t = np.linspace(time[0], time[-1], num=1000, endpoint=True)
    # ax3.plot(t, F_p_back(t), c='tab:red')
    ax3.plot(time, [BW*model.F_p[fe,ncp,'back_knee'].value for fe in model.fe], c='tab:red')
    ax3.set_ylabel('Force (N)')
    fig.tight_layout()
    if savefig: fig.savefig(f'figures/{description}-actuators-plot.pdf', bbox_inches='tight')


def plot_GRF(m):
    fig, ax = plt.subplots(2, sharex=True)
    time, time_cp = get_time_arrays(m)
    nfe, ncp = len(m.fe), len(m.cp)
    ax[0].set_title('Front foot')
    ax[0].plot(time_cp, [BW*m.GRFy[fe,cp,'front_foot'].value for fe in m.fe for cp in m.cp], color='tab:red')
    ax[0].set_ylabel('Force (N)')
    ax0 = ax[0].twinx()
    ax0.plot(time_cp, [m.foot_y[fe,cp,'front_foot'].value for fe in m.fe for cp in m.cp])
    ax0.set_ylabel('Foot height (m)')
    ax[1].set_title('Back foot')
    ax[1].plot(time_cp, [BW*m.GRFy[fe,cp,'back_foot'].value for fe in m.fe for cp in m.cp], color='tab:red')
    ax[1].set_ylabel('Force (N)')
    ax1 = ax[1].twinx()
    ax1.plot(time_cp, [m.foot_y[fe,cp,'back_foot'].value for fe in m.fe for cp in m.cp])
    ax1.set_ylabel('Foot height (m)')
    fig.tight_layout()


def plot_footfall(m):
    fig, ax = plt.subplots()
    nfe, ncp = len(m.fe), len(m.cp)
    time, time_cp = get_time_arrays(m)
    ax.set_title('Footfall')
    # plot front foot
    front_contact = np.array([m.GRFy[fe,ncp,'front_foot'].value for fe in m.fe]) > 0
    ax.eventplot(time[front_contact], colors='k', lineoffsets=0, linelengths=2, linewidths=8)
    # plot back foot
    front_contact = np.array([m.GRFy[fe,ncp,'back_foot'].value for fe in m.fe]) > 0
    ax.eventplot(time[front_contact], colors='k', lineoffsets=2, linelengths=2, linewidths=8)
    ax.set_xlim([time[0], time[-1]])
    ax.set_yticks([0, 2])
    ax.set_yticklabels(['Front', 'Back'])
    ax.set_xlabel('time (s)')
    fig.tight_layout()


def plot_timing(m):
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})
    nfe, ncp = len(m.fe), len(m.cp)
    time, time_cp = get_time_arrays(m)
    # ax.scatter(time, np.zeros(len(time)))
    timesteps = [m.h[fe].value * m.hm * 1000 for fe in m.fe]
    ax[0].set_title('Timestep Variation')
    ax[0].plot(range(1, len(timesteps)+1), timesteps)
    ax[0].axhline(y=m.hm*m.h[1].ub*1000, color='r', linestyle='-')
    ax[0].axhline(y=m.hm*m.h[1].lb*1000, color='r', linestyle='-')
    ax[0].set_ylabel('Δt (ms)')
    ax[0].set_xlabel('Node')
    ax[1].set_title('Node Spacing')
    ax[1].scatter(time, np.zeros(len(time)))
    ax[1].set_yticks([])
    ax[1].set_xlabel('Time (s)')
    fig.tight_layout()


if __name__ == '__main__':
    import os
    import sys
    import cloudpickle
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # loading saved model
    with open('./kemba/cache/solution.pkl', mode='rb') as file:
        model = cloudpickle.load(file)
    print_stats(model)
    animate(model, speed=1.0, follow=False)
    # plot_timing(model)
    # plot_footfall(model)
    # plot_penalties(model)
    plot_actuators(model)
    # plot_GRF(model)
    # stop_motion(model, n_frames=6)
    
    fig, ax = plt.subplots(3)
    time, time_cp = get_time_arrays(model)
    nfe, ncp = len(model.fe), len(model.cp)
    
    # # rebound torques
    # ax[0].plot(time, [BW*(model.τ_r[n,1,'front_hip','+'].value - model.τ_r[n,1,'front_hip','-'].value) for fe in nodes])
    # ax[1].plot(time, [BW*(model.τ_r[n,1,'back_hip','+'].value - model.τ_r[n,1,'back_hip','-'].value) for fe in nodes])
    # ax[2].plot(time, [BW*(model.τ_r[n,1,'front_knee','+'].value - model.τ_r[n,1,'front_knee','-'].value) for fe in nodes])
    # ax[3].plot(time, [BW*(model.τ_r[n,1,'back_knee','+'].value - model.τ_r[n,1,'back_knee','-'].value) for fe in nodes])
    
    # # GRF check plot
    # ax[0].set_title('Foot height')
    # ax[0].plot(time, [model.foot_y[n,1,'front_foot'].value for fe in model.fe])
    # ax[1].plot(time, [model.foot_y[n,1,'back_foot'].value for fe in model.fe])
    # ax[2].set_title('Ground reaction force')
    # ax[2].plot(time, [BW*model.GRFy[n,1,'front_foot'].value for fe in model.fe])
    # ax[3].plot(time, [BW*model.GRFy[n,1,'back_foot'].value for fe in model.fe])

    # # ground reactions
    # ax[0].plot(time, [BW*model.GRFy[n,1,'front_foot'].value for fe in model.fe])
    # ax[1].plot(time, [BW*(model.GRFx[n,1,'front_foot','+'].value - model.GRFx[n,1,'front_foot','-'].value) for fe in model.fe])
    # ax[2].plot(time, [BW*model.GRFy[n,1,'back_foot'].value for fe in model.fe])
    # ax[3].plot(time, [BW*(model.GRFx[n,1,'back_foot','+'].value - model.GRFx[n,1,'back_foot','-'].value) for fe in model.fe])

    # # actuator torques and forces
    # ax[0].plot(time, [BW*model.τ_m[n,1,'front_hip'].value for fe in model.fe])
    # ax[1].plot(time, [BW*model.τ_m[n,1,'back_hip'].value for fe in model.fe])
    # ax[2].plot(time, [BW*model.F_p[n,1,'front_knee'].value for fe in model.fe])
    # ax[3].plot(time, [BW*model.F_p[n,1,'back_knee'].value for fe in model.fe])

    # # constraint forces
    # ax[0].plot(time, [BW*model.F_c[fe,1,'x','front_knee'].value for fe in model.fe])
    # ax[1].plot(time, [BW*model.F_c[fe,1,'y','front_knee'].value for fe in model.fe])
    # ax[2].plot(time, [BW*model.F_c[fe,1,'x','back_knee'].value for fe in model.fe])
    # ax[3].plot(time, [BW*model.F_c[fe,1,'y','back_knee'].value for fe in model.fe])

    # # piston length
    # ax[0].plot(time, [model.q[n,cp,'l5'].value for fe in model.fe])
    # ax[1].plot(time, [model.q[n,cp,'l6'].value for fe in model.fe])
    # ax[2].plot(time, [model.dq[n,cp,'l5'].value for fe in model.fe])
    # ax[3].plot(time, [model.dq[n,cp,'l6'].value for fe in model.fe])

    # piston length
    # ax[0].plot(time_cp, [model.dq[fe,cp,'l5'].value for fe in model.fe for cp in model.cp])
    # ax[1].plot(time_cp, [model.dq[fe,cp,'l6'].value for fe in model.fe for cp in model.cp])
    # ax[2].plot(time_cp, [model.ddq[fe,cp,'y0'].value for fe in model.fe for cp in model.cp])
    # ax[3].scatter(time, np.zeros(len(model.h)))

    # # foot height
    # ax[0].plot(time, [model.foot_y[n,cp,'front_foot'].value for fe in model.fe])
    # ax[1].plot(time, [model.foot_y[n,cp,'back_foot'].value for fe in model.fe])

    # joint angles
    ax[0].plot(time, [model.dq[fe,1,'x0'].value for fe in model.fe])
    ax[1].plot(time, [model.q[fe,1,'y0'].value for fe in model.fe])
    ax[2].plot(time, [model.dq[fe,1,'θ0'].value for fe in model.fe])
    # ax[3].plot(time, [model.joint_θ[fe,cp,'back_knee'].value for fe in model.fe])

    # # # body state
    # ax[0].plot(time, [model.q[fe,1,'x0'].value for fe in model.fe])
    # ax[1].plot(time, [model.dq[fe,1,'x0'].value for fe in model.fe])
    # ax[2].plot(time, [model.ddq[fe,1,'x0'].value for fe in model.fe])

    fig.tight_layout()
    plt.show() # call after creating plots
