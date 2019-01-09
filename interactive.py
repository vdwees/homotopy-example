import itertools
import time

import casadi as ca
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import ColumnDataSource, CustomJS, LinearInterpolator, Slider
from bokeh.plotting import figure, output_file, show
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

# These parameters correspond to Table 1
T = 72
dt = 10 * 60
times = np.arange(0, (T + 1) * dt, dt)
H_b = -3.0
l = 10000.0
w = 50.0
C = 40.0
H_nominal = 0.0
Q_nominal = 100
Q0 = ca.DM([100, 100])
H0 = ca.DM([0, 0])  # todo: fixme

# Generic constants
g = 9.81
eps = 1e-12

# Derived quantities
dx = l / 2.0
A_nominal = w * (H_nominal - H_b)
P_nominal = w + 2 * (H_nominal - H_b)

# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)

# Symbols
Q = ca.MX.sym("Q", 2, T)
H = ca.MX.sym("H", 2, T)
theta = ca.MX.sym("theta")

# Left boundary condition
Q_left = np.full(T + 1, 100)
Q_left[T // 4 : T // 2] = 300.0
Q_left = ca.DM(Q_left).T

# Hydraulic constraints
Q_full = ca.vertcat(Q_left, ca.horzcat(Q0, Q))
H_full = ca.horzcat(H0, H)
A_full = w * 0.5 * (H_full[1:, :] + H_full[:-1, :] - 2 * H_b)
P_full = w + (H_full[1:, :] + H_full[:-1, :] - 2 * H_b)

c = w * (H_full[:, 1:] - H_full[:, :-1]) / dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dx
d = (
    (Q_full[1:-1, 1:] - Q_full[1:-1, :-1]) / dt
    + g
    * (theta * A_full[:, 1:] + (1 - theta) * A_nominal)
    * (H_full[1:, 1:] - H_full[:-1, 1:])
    / dx
    + g
    * (
        theta * P_full[:, :-1] * sabs(Q_full[1:-1, 1:]) / A_full[:, :-1] ** 2
        + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
    )
    * Q_full[1:-1, 1:]
    / C ** 2
)

# Objective function
f = ca.sum1(ca.vec(H[:, 1:] ** 2))

# Variable bounds
lbQ = ca.repmat(ca.DM([-np.inf, 0.0]), 1, T)
ubQ = ca.repmat(ca.DM([np.inf, 200.0]), 1, T)
ubQ[1, 0] = 100
lbH = ca.repmat(H_b + 2.0, 2, T)
ubH = ca.repmat(np.inf, 2, T)

# Optimization problem
assert Q.size() == lbQ.size()
assert Q.size() == ubQ.size()
assert H.size() == lbH.size()
assert H.size() == ubH.size()

X = ca.veccat(Q, H)
lbX = ca.veccat(lbQ, lbH)
ubX = ca.veccat(ubQ, ubH)

g = ca.veccat(c, d)
lbg = ca.repmat(0, g.size1())
ubg = lbg

nlp = {"f": f, "g": g, "x": X, "p": theta}
solver = ca.nlpsol(
    "nlpsol",
    "ipopt",
    nlp,
    {
        "ipopt": {
            "tol": 1e-10,
            "constr_viol_tol": 1e-10,
            "acceptable_tol": 1e-10,
            "acceptable_constr_viol_tol": 1e-10,
            "print_level": 0,
            "print_timing_statistics": "no",
            "fixed_variable_treatment": "make_constraint",
        }
    },
)

# Initial guess
x0 = ca.repmat(0, X.size1())

# Solve
t1 = time.time()

raw_results = {}

theta_values = np.linspace(0.0, 1.0, 30)
variable_names = "H_1", "H_2", "Q_1", "Q_2", "Q_3"
results = {var: {theta: None for theta in theta_values} for var in variable_names}
for theta_value in theta_values:
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x0 = solution["x"]
    Q_res = ca.reshape(x0[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x0[Q.size1() * Q.size2() :], H.size1(), H.size2())
    results["Q_1"][theta_value] = np.array(Q_left).flatten()
    results["Q_2"][theta_value] = np.array(ca.horzcat(Q0[0], Q_res[0, :])).flatten()
    results["Q_3"][theta_value] = np.array(ca.horzcat(Q0[1], Q_res[1, :])).flatten()
    results["H_1"][theta_value] = np.array(ca.horzcat(H0[0], H_res[0, :])).flatten()
    results["H_2"][theta_value] = np.array(ca.horzcat(H0[1], H_res[1, :])).flatten()

shared_data = {"times": times, "theta_values": [theta_values for time in times]}
plots = []
sliders = []
for var in variable_names:
    plot_data = shared_data.copy()
    plot_data["y"] = results[var][1.0]
    plot_data["all_results"] = [
        [results[var][theta][tidx] for theta in theta_values]
        for tidx, time in enumerate(times)
    ]
    source = ColumnDataSource(data=plot_data)
    plot = figure(
        title=var,
        plot_width=400,
        plot_height=400,
        y_range=(
            min(np.min(results[var][theta]) for theta in theta_values),
            max(np.max(results[var][theta]) for theta in theta_values),
        ),
    )
    plot.line("times", "y", source=source, line_width=3, line_alpha=0.6)
    plots.append(plot)

    with open("interactive.js", "r") as code_file:
        callback = CustomJS(args=dict(source=source), code=code_file.read())

    slider = Slider(
        start=0.0, end=1.0, value=1, step=0.01, title="theta", callback=callback
    )
    sliders.append(slider)

elements = [column(s, p) for s, p in zip(sliders, plots)]
layout = gridplot(elements, ncols=3)
output_file("interactive.html", title="Homotopy Deformation")

show(layout)
