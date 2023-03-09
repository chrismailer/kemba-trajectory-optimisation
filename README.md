# Kemba Trajectory Optimisation

Code to generate optimal trajectories for [Kemba](https://youtu.be/u2Hn26uojoM) - a novel planar legged robot which uses a combination of electric and pneumatic actuation.

<p align="middle">
  <img src="/figures/rigid-bodies.png" height="300" />
  <img src="/figures/kemba.jpeg" height="300" />
</p>

<p align="middle">
  <img src="/figures/backflip-frames-plot.jpg" width="600" />
</p>

## Getting Started
1. [Install Ipopt](https://github.com/African-Robotics-Unit/docs/blob/main/linear-solvers.md)
2. Install required python dependencies with `pip install -r requirements.txt`
3. Run `optimise.py`

## File Structure
    .
    ├── ...
    ├── core                    # Files not specific to robot or optimisation formulation
    │   ├── actuators.py        # Motor and piston properties and behaviours
    │   ├── collocation.py      # Collocation methods
    │   └── utils.py            # Solver function and parameters
    ├── figures                 # Folder for figures from visualise.py
    │   └── ...
    ├── kemba                   # File specific to robot and optimisation formulation
    │   ├── cache               # Folder for cached files
    |   │   ├── dynamics.pkl    # Cached dynamics
    |   │   └── solution.pkl    # Cached optimisation solution
    │   ├── constants.py        # Robot and boom constants
    │   ├── dynamics.py         # Calculate, simplify and cache dynamics equations
    │   ├── model.py            # Robot model pyomo formulation
    │   └── tasks.py            # Tasks pyomo formulation
    ├── export.py               # Export optimisation solution to .csv file
    ├── optimise.py             # Run trajectory optimisation
    └── requirements.txt        # Python dependencies

