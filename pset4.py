import numpy as np
import matplotlib.pyplot as plt

# ===== Define PDE and Numerical Methods =====
# PDE: u_t + u_x = 0
# Boundary Condition (BC): Periodic
# Initial Condition (IC): u(0, x) = sin(x)
L = 2 * np.pi  # Length of the domain (periodic boundary)
N = 100        # (a) Number of divisions in the x-direction

dx = L / N  # Spatial step size
x = np.linspace(0, L, N, endpoint=False)  # Spatial grid points
u0 = np.sin(x)  # Initial condition: u(0, x) = sin(x)

# Function to calculate the spatial derivative using central differences with periodic boundary
def spatial_derivative(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

# Function to calculate the right-hand side of the PDE (flux)
def f(u, t, dx):
    return -spatial_derivative(u, dx)

# Single RK4 step function for time integration
def rk4_step(u, t, dt, dx):
    k1 = f(u, t, dx)
    k2 = f(u + 0.5 * dt * k1, t + 0.5 * dt, dx)
    k3 = f(u + 0.5 * dt * k2, t + 0.5 * dt, dx)
    k4 = f(u + dt * k3, t + dt, dx)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Function to solve the PDE using RK4 over the given time span
def solve(u_initial, dt, T, dx):
    steps = int(T / dt)  # Number of time steps
    u = u_initial.copy()  # Copy the initial condition
    t = 0.0  # Initialize time
    for _ in range(steps):
        u = rk4_step(u, t, dt, dx)  # Perform RK4 time integration
        t += dt  # Update time
    return u, t

# ===== (c) Test with varying dt =====
dts = [0.1, 0.05, 0.01, dx, dx / 2, dx / 4, dx / 8, dx / 20]  # Different time step sizes
T_values = [1.0, 10.0]  # Time intervals for testing

# Store solutions for each combination of T and dt
solutions = {}
for T in T_values:
    solutions[T] = {}
    for dt_test in dts:
        u_approx, _ = solve(u0, dt_test, T, dx)  # Solve PDE for given dt and T
        solutions[T][dt_test] = u_approx

# ===== (d) Plot results for T=1 and T=10 =====
plt.figure(figsize=(10, 6))
for i, T in enumerate(T_values, start=1):
    plt.subplot(2, 1, i)
    for dt_test in dts:
        plt.plot(x, solutions[T][dt_test], label=f"dt={dt_test:.5f}")
    plt.title(f"u(x) at t={T}")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
plt.tight_layout()
plt.show()

# ===== (e) Reference Solution and Error Analysis =====
# Reference solution: Use a very small dt (dt_ref = dx / 20)
dt_ref = dx / 20
T_ref = 1.0
u_ref, _ = solve(u0, dt_ref, T_ref, dx)  # Solve PDE with reference dt

# Compute solutions for various dt and compare with reference solution
# Error metrics: L_inf (max norm) and L2 norm
def l_inf_error(u_test, u_ref):
    return np.max(np.abs(u_test - u_ref))

def l2_error(u_test, u_ref, dx):
    diff = u_test - u_ref
    return np.sqrt(np.sum(diff**2) * dx)

test_dts = [dx, dx / 2, dx / 4, dx / 8, dx / 16]  # Time steps for testing

# Compute errors for each dt
e_inf_list = []
e_l2_list = []
for dt_test in test_dts:
    u_test, _ = solve(u0, dt_test, T_ref, dx)  # Solve PDE with the current dt
    e_inf = l_inf_error(u_test, u_ref)  # Compute L_inf error
    e_l2 = l2_error(u_test, u_ref, dx)  # Compute L2 error
    e_inf_list.append(e_inf)
    e_l2_list.append(e_l2)

# Plot error vs dt on a log-log scale
plt.figure()
plt.loglog(test_dts, e_inf_list, 'o-', label='L_inf error')
plt.loglog(test_dts, e_l2_list, 's-', label='L2 error')
plt.xlabel('dt')
plt.ylabel('Error')
plt.title('Error vs dt (log-log)')
plt.grid(True)
plt.legend()
plt.show()

# Compute error ratios and log2(e1/e2) for successive dt values
def log2_ratio(e1, e2):
    return np.log2(e1 / e2)

print("L_inf error ratios in log2 scale:")
for i in range(len(e_inf_list) - 1):
    ratio = log2_ratio(e_inf_list[i], e_inf_list[i + 1])
    print(f"log2(e_inf(dt={test_dts[i]:.5f}) / e_inf(dt={test_dts[i + 1]:.5f})) = {ratio:.4f}")

print("\nL2 error ratios in log2 scale:")
for i in range(len(e_l2_list) - 1):
    ratio = log2_ratio(e_l2_list[i], e_l2_list[i + 1])
    print(f"log2(e_l2(dt={test_dts[i]:.5f}) / e_l2(dt={test_dts[i + 1]:.5f})) = {ratio:.4f}")