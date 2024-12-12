import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
mu = 0.012277471
nu = 1 - mu
t_final = 17.065211656

# System of ODEs
def equations(t, z):
    z1, z2, z3, z4 = z
    D1 = ((z1 + mu)**2 + z3**2)**1.5
    D2 = ((z1 - nu)**2 + z3**2)**1.5

    dz1_dt = z2
    dz2_dt = z1 + 2*z4 - nu * (z1 + mu) / D1 - mu * (z1 - nu) / D2
    dz3_dt = z4
    dz4_dt = z3 - 2*z2 - nu * z3 / D1 - mu * z3 / D2

    return [dz1_dt, dz2_dt, dz3_dt, dz4_dt]

# Initial conditions
z0 = [0.994, 0, 0, -2.0015851063790825]

# Solve using solve_ivp
solution = solve_ivp(equations, [0, t_final], z0, method='RK45', rtol=1e-9, atol=1e-9)

# Extract solution
z1, z2, z3, z4 = solution.y

# Plot the orbit
plt.figure(figsize=(8, 6))
plt.plot(z1, z3, label='Orbit')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (solve_ivp)')
plt.grid()
plt.legend()
plt.show()
plt.savefig('./pset3_RK45.png')


# Euler's method
def euler_method(equations, t_span, z0, n_steps):
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    z = np.zeros((len(z0), len(t)))
    z[:, 0] = z0

    for i in range(n_steps):
        z[:, i + 1] = z[:, i] + dt * np.array(equations(t[i], z[:, i]))

    return t, z

n_steps_euler = 24000
t_euler, z_euler = euler_method(equations, [0, t_final], z0, n_steps_euler)

# Plot Euler's method solution
plt.figure(figsize=(8, 6))
plt.plot(z_euler[0], z_euler[2], label='Orbit (Euler)')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (Euler Method)')
plt.grid()
plt.legend()
plt.show()
plt.savefig('./pset3_euler.png')

# Runge-Kutta 4th order
def rk4_method(equations, t_span, z0, n_steps):
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    z = np.zeros((len(z0), len(t)))
    z[:, 0] = z0

    for i in range(n_steps):
        k1 = np.array(equations(t[i], z[:, i]))
        k2 = np.array(equations(t[i] + dt / 2, z[:, i] + dt / 2 * k1))
        k3 = np.array(equations(t[i] + dt / 2, z[:, i] + dt / 2 * k2))
        k4 = np.array(equations(t[i] + dt, z[:, i] + dt * k3))
        z[:, i + 1] = z[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, z

n_steps_rk4 = 6000
t_rk4, z_rk4 = rk4_method(equations, [0, t_final], z0, n_steps_rk4)

# Plot RK4 method solution
plt.figure(figsize=(8, 6))
plt.plot(z_rk4[0], z_rk4[2], label='Orbit (RK4)')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (RK4 Method)')
plt.grid()
plt.legend()
plt.show()
plt.savefig('./pset3_RK4.png')

print("Number of steps used by solve_ivp:", len(solution.t))

t_final_3 = 3 * t_final
solution_3 = solve_ivp(equations, [0, t_final_3], z0, method='RK45', rtol=1e-12, atol=1e-12)

# Plot 3-period orbit
plt.figure(figsize=(8, 6))
plt.plot(solution_3.y[0], solution_3.y[2], label='Orbit (3 Periods)')
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object over 3 Periods (solve_ivp)')
plt.grid()
plt.legend()
plt.show()
plt.savefig('./pset3_3_periods.png')