from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.environ import sqrt
import numpy as np

# '/usr/local/bin/ipopt' on my mac

# https://github.com/alknemeyer/physical_education/blob/328724c5f8b0bc08351c99539d9320116fa4a822/physical_education/utils.py#L428
# https://coin-or.github.io/Ipopt/OPTIONS.html

def default_solver(path, warm_start=False, approximate_hessian=False):
    opt = SolverFactory('ipopt', executable=path)
    # solver options
    opt.options["linear_solver"] = 'ma86'
    opt.options["expect_infeasible_problem"] = 'yes' # Enable heuristics to quickly detect an infeasible problem.
    # opt.options["linear_system_scaling"] = 'none'
    # opt.options["mu_strategy"] = "adaptive"
    opt.options['halt_on_ampl_error'] = 'yes'
    opt.options['OF_ma86_scaling'] = 'none' # default is 'mc64'
    opt.options['OF_hessian_approximation'] = 'limited-memory' if approximate_hessian else 'exact' # less accurate but much faster method to compute the Hessian
    opt.options['OF_print_timing_statistics'] = 'yes'
    opt.options['OF_warm_start_init_point'] = 'yes' if warm_start else 'no'
    opt.options['OF_acceptable_tol'] = 1e-6  # default: 1e-6
    opt.options["print_level"] = 5 # prints a log with each iteration (you want to this - it's the only way to see progress.)
    opt.options["max_iter"] = 10_000 # maximum number of iterations
    opt.options["max_cpu_time"] = 10_000 # maximum cpu time in seconds
    opt.options["Tol"] = 1e-6 # the tolerance for feasibility. Considers constraints satisfied when they're within this margin.
    opt.options['output_file'] = './ipopt.log'
    return opt


def get_vals_v(var, idxs):
    """
    Verbose version that doesn't try to guess stuff for ya. Usage:
    >>> get_vals(m.q, (m.N, m.DOF))
    """
    m = var.model
    arr = np.array([var[idx].value for idx in var]).astype(float)
    return arr.reshape(*(len(i) for i in idxs))
