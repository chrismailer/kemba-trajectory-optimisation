from kemba.model import create_model, penalty, torque_squared, time_sum, piston_penalty_sum
import kemba.tasks as tasks
from core.utils import default_solver
from core.collocation import implicit_euler, radau_2, radau_3
from pyomo.opt import TerminationCondition
import cloudpickle as pickle
from pyomo.environ import Objective, minimize

#! create model
model = create_model(duration=0.4, nfe=50, ncp=1, integrator=implicit_euler, timestep_bounds=(0.8,1.2))
#! define task
# tasks.periodic_bounding(model, stride=1.0) # tasks.periodic_bounding(model, 1.4)
# timings = {'front_foot':(0.2, 0.4), 'back_foot':(0.6, 0.8)} # timings = {'front_foot':(0.2, 0.4), 'back_foot':(0.6, 0.8)}
# tasks.prescribe_contact_order(model, timings, min_foot_height=1e-3, min_GRFy=1e-3)
tasks.initiate_bounding(model, speed=2)
# tasks.terminate_bounding(model)
# tasks.popup(model)
# tasks.jump(model, height=1.0)
# tasks.jump_and_land(model, height=0.8)
# tasks.jump_forwards(model, distance=2.0)
# tasks.stand(model)
# tasks.backflip(model)
# tasks.drop_test(model, 1.0)
#! define cost function
model.cost = Objective(expr=1e4*penalty(model) + 1e2*piston_penalty_sum(model) + torque_squared(model), sense=minimize)
#! solving
result = default_solver('/usr/local/bin/ipopt', approximate_hessian=False).solve(model, tee=True)
if result.solver.termination_condition == TerminationCondition.optimal:
    with open('./kemba/cache/solution.pkl', mode='wb') as file:
        pickle.dump(model, file)
else:
    quit()
