# -*- coding: utf-8 -*-
"""BatchMPC1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1c4b78YnOmHJ3jPECekLtSGfievFBOMkn
"""

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# System matrices
A = np.array([[1.0, 1.0],
              [0.0, 1.0]])
B = np.array([[0.5], [1.0]])
nx, nu = A.shape[0], B.shape[1]

# Cost function weights
Q = np.eye(nx)  # State tracking cost
R = np.eye(nu)  # Control effort cost
P = Q  # Terminal cost

# Constraints
xL = np.array([-10, -10])  # State lower bounds
xU = np.array([10, 10])    # State upper bounds
uL = np.array([-2])        # Control lower bound
uU = np.array([2])         # Control upper bound

# Time horizon and initial conditions
N = 10  # Prediction horizon
delta_t = 1.0
x0 = np.array([5.0, 0.0])  # Initial state
x_target = np.array([0.0, 0.0])  # Target state
Lsim = 30  # Simulation length

# Batch optimization function
def batch_optimization(A, B, P, Q, R, N, x0, xL, xU, uL, uU):
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(A, 0)
    model.nu = np.size(B, 1)

    # Sets
    model.tIDX = pyo.Set(initialize=range(model.N+1), ordered=True)
    model.xIDX = pyo.Set(initialize=range(model.nx), ordered=True)
    model.uIDX = pyo.Set(initialize=range(model.nu), ordered=True)

    # Variables
    model.x = pyo.Var(model.xIDX, model.tIDX, bounds=lambda m, i, t: (xL[i], xU[i]))
    model.u = pyo.Var(model.uIDX, model.tIDX, bounds=lambda m, i, t: (uL[i], uU[i]))

    # Objective
    def objective_rule(model):
        costX = sum((model.x[i, t] - x_target[i])**2 * Q[i, i] for t in model.tIDX for i in model.xIDX)
        costU = sum((model.u[j, t])**2 * R[j, j] for t in range(model.N) for j in model.uIDX)
        costTerminal = sum((model.x[i, model.N] - x_target[i])**2 * P[i, i] for i in model.xIDX)
        return costX + costU + costTerminal

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Dynamics constraints
    def dynamics_rule(model, i, t):
        if t < model.N:
            return model.x[i, t+1] == sum(A[i, j] * model.x[j, t] for j in model.xIDX) + \
                                          sum(B[i, j] * model.u[j, t] for j in model.uIDX)
        return pyo.Constraint.Skip

    model.dynamics = pyo.Constraint(model.xIDX, model.tIDX, rule=dynamics_rule)

    # Initial state
    def init_rule(model, i):
        return model.x[i, 0] == x0[i]

    model.init_constraints = pyo.Constraint(model.xIDX, rule=init_rule)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    # Extract results
    xref = np.array([[pyo.value(model.x[i, t]) for t in model.tIDX] for i in model.xIDX])
    uref = np.array([[pyo.value(model.u[i, t]) for t in range(model.N)] for i in model.uIDX])

    return xref, uref

# MPC function
def solve_cftoc(A, B, P, Q, R, N, xk, xL, xU, uL, uU, xref, uref):
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = A.shape[0]
    model.nu = B.shape[1]

    # Sets
    model.tIDX = pyo.Set(initialize=range(N+1), ordered=True)
    model.xIDX = pyo.Set(initialize=range(model.nx), ordered=True)
    model.uIDX = pyo.Set(initialize=range(model.nu), ordered=True)

    # Variables
    model.x = pyo.Var(model.xIDX, model.tIDX, bounds=lambda m, i, t: (xL[i], xU[i]))
    model.u = pyo.Var(model.uIDX, model.tIDX, bounds=lambda m, i, t: (uL[i], uU[i]))

    # Objective
    def objective_rule(model):
        costX = sum((model.x[i, t] - xref[i, t])**2 * Q[i, i] for t in model.tIDX for i in model.xIDX)
        costU = sum((model.u[j, t] - uref[j, t])**2 * R[j, j] for t in range(model.N) for j in model.uIDX)
        costTerminal = sum((model.x[i, model.N] - xref[i, model.N])**2 * P[i, i] for i in model.xIDX)
        return costX + costU + costTerminal

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Dynamics constraints
    def dynamics_rule(model, i, t):
        if t < N:
            return model.x[i, t+1] == sum(A[i, j] * model.x[j, t] for j in model.xIDX) + \
                                          sum(B[i, j] * model.u[j, t] for j in model.uIDX)
        return pyo.Constraint.Skip

    model.dynamics = pyo.Constraint(model.xIDX, model.tIDX, rule=dynamics_rule)

    # Initial state constraint
    def init_rule(model, i):
        return model.x[i, 0] == xk[i]

    model.init_constraints = pyo.Constraint(model.xIDX, rule=init_rule)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)

    xOpt = np.array([[pyo.value(model.x[i, t]) for t in model.tIDX] for i in model.xIDX])
    uOpt = np.array([[pyo.value(model.u[i, t]) for t in range(N)] for i in model.uIDX])
    return model, results.solver.termination_condition == 'optimal', xOpt, uOpt

# Main simulation loop
xref, uref = batch_optimization(A, B, P, Q, R, N, x0, xL, xU, uL, uU)
xk = x0
states = [xk]
for t in range(Lsim):
    xref_preview = xref[:, t:t+N+1]
    uref_preview = uref[:, t:t+N]
    model, feas, xOpt, uOpt = solve_cftoc(A, B, P, Q, R, N, xk, xL, xU, uL, uU, xref_preview, uref_preview)
    if not feas:
        print("Problem is infeasible!")
        break
    xk = A @ xk + B @ uOpt[:, 0]
    states.append(xk)

# Plot results
states = np.array(states)
plt.plot(states[:, 0], states[:, 1], label="Trajectory")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()