import itertools
import time

import casadi as ca
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

# These parameters correspond to Table 1
T = 10
dt = 5 * 60
times = np.arange(0, (T + 1) * dt, dt)
H_b = -1.0
l = 1000.0
w = 5.0
C = 10.0
H_nominal = -0.25
Q_nominal = 0.5
Q0 = ca.DM([0, 0])
H0 = ca.DM([0, 0])

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
Q_left = np.zeros(T + 1)
Q_left[T // 3 : 2 * (T // 3)] = 2.0
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
ubQ = ca.repmat(ca.DM([np.inf, 1.0]), 1, T)
lbH = ca.repmat(H_b, 2, T)
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

results = {}

theta_values = np.linspace(0.0, 1.0, 70)
variable_names = "H_1", "H_2", "Q_1", "Q_2", "Q_3"

for theta_value in theta_values:
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x0 = solution["x"]
    Q_res = ca.reshape(x0[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x0[Q.size1() * Q.size2() :], H.size1(), H.size2())
    d = {}
    d["Q_1"] = np.array(Q_left).flatten()
    d["Q_2"] = np.array(ca.horzcat(Q0[0], Q_res[0, :])).flatten()
    d["Q_3"] = np.array(ca.horzcat(Q0[1], Q_res[1, :])).flatten()
    d["H_1"] = np.array(ca.horzcat(H0[0], H_res[0, :])).flatten()
    d["H_2"] = np.array(ca.horzcat(H0[1], H_res[1, :])).flatten()
    assert set(variable_names) == set(d.keys())
    results[theta_value] = d

t2 = time.time()

print("Time elapsed in solver: {}s".format(t2 - t1))


# Use greyscale style for plots
plt.style.use("grayscale")

suffix = "pdf"  # "png"

# Generate Aggregated Plot
n_subplots = 2
width = 4
height = 3
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(width, height))

for theta, var in itertools.product(theta_values[-1:], variable_names):
    axarr[0 if var.startswith("Q") else 1].step(
        times, results[theta][var], where="mid", label=f"{var}"
    )

axarr[0].set_ylabel("Flow Rate [m³/s]")
axarr[1].set_ylabel("Water Level [m]")
axarr[1].set_xlabel("Time [s]")

plt.autoscale(enable=True, axis="x", tight=True)

# Shrink margins
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])
    axarr[i].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8}
    )

# Output Plot
plt.savefig(f"final_results.{suffix}")

# Generate Individual Deformation Plots
width = 4
height = 2
for var in variable_names:
    fig, ax = plt.subplots(1, figsize=(width, height))
    ax.set_title(var)
    for theta in theta_values:
        ax.step(
            times,
            results[theta][var],
            where="mid",
            color=str(0.9 - theta * 0.9),
            linewidth=2,
        )

    # Shrink margins
    ax.set_ylabel("Flow Rate [m³/s]" if var.startswith("Q") else "Water Level [m]")
    ax.set_xlabel("Time [s]")

    fig.tight_layout()

    # Output Plot
    plt.savefig(f"{var}.{suffix}")

# Generate a Bar Scale Legend
width = 7
height = 1
fig, ax = plt.subplots(1, figsize=(width, height))

cmap = ListedColormap(list(map(str, np.linspace(1.0, 0.1, 256))))
norm = Normalize(vmin=0.0, vmax=1.0)
cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_label("Homotopy Parameter")

fig.tight_layout()

plt.savefig(f"colorbar.{suffix}")
