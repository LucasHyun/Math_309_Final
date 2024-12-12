import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for the restricted three-body problem
mu = 0.012277471  # Gravitational parameter for the smaller primary
nu = 1 - mu       # Gravitational parameter for the larger primary
t_final = 17.065211656  # Final time for the integration (one period)

# System of ODEs defining the equations of motion
def equations(t, z):
    z1, z2, z3, z4 = z  # Unpack the state variables
    D1 = ((z1 + mu)**2 + z3**2)**1.5  # Distance to the smaller primary
    D2 = ((z1 - nu)**2 + z3**2)**1.5  # Distance to the larger primary

    # Equations of motion
    dz1_dt = z2
    dz2_dt = z1 + 2*z4 - nu * (z1 + mu) / D1 - mu * (z1 - nu) / D2
    dz3_dt = z4
    dz4_dt = z3 - 2*z2 - nu * z3 / D1 - mu * z3 / D2

    return [dz1_dt, dz2_dt, dz3_dt, dz4_dt]

# Initial conditions for the problem
z0 = [0.994, 0, 0, -2.0015851063790825]  # Initial positions and velocities

# Solve the system using solve_ivp (Runge-Kutta method)
solution = solve_ivp(equations, [0, t_final], z0, method='RK45', rtol=1e-9, atol=1e-9)

# Extract the solution for plotting
z1, z2, z3, z4 = solution.y

# Plot the orbit calculated by solve_ivp
plt.figure(figsize=(8, 6))
plt.plot(z1, z3, label='Orbit')  # Plot position in the y1-y2 plane
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (solve_ivp)')
plt.grid()  # Add grid for better visualization
plt.legend()  # Add legend to the plot
plt.show()
plt.savefig('./pset3_RK45.png')  # Save the plot to a file

# Euler's method for solving ODEs
def euler_method(equations, t_span, z0, n_steps):
    t0, tf = t_span  # Extract start and end times
    dt = (tf - t0) / n_steps  # Compute the time step size
    t = np.linspace(t0, tf, n_steps + 1)  # Time array
    z = np.zeros((len(z0), len(t)))  # Array to store the solution
    z[:, 0] = z0  # Set initial conditions

    for i in range(n_steps):
        z[:, i + 1] = z[:, i] + dt * np.array(equations(t[i], z[:, i]))  # Euler update

    return t, z

# Number of steps for Euler's method
n_steps_euler = 24000
# Solve the system using Euler's method
t_euler, z_euler = euler_method(equations, [0, t_final], z0, n_steps_euler)

# Plot the orbit calculated by Euler's method
plt.figure(figsize=(8, 6))
plt.plot(z_euler[0], z_euler[2], label='Orbit (Euler)')  # Plot position in the y1-y2 plane
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (Euler Method)')
plt.grid()  # Add grid for better visualization
plt.legend()  # Add legend to the plot
plt.show()
plt.savefig('./pset3_euler.png')  # Save the plot to a file

# Runge-Kutta 4th-order method for solving ODEs
def rk4_method(equations, t_span, z0, n_steps):
    t0, tf = t_span  # Extract start and end times
    dt = (tf - t0) / n_steps  # Compute the time step size
    t = np.linspace(t0, tf, n_steps + 1)  # Time array
    z = np.zeros((len(z0), len(t)))  # Array to store the solution
    z[:, 0] = z0  # Set initial conditions

    for i in range(n_steps):
        k1 = np.array(equations(t[i], z[:, i]))  # First slope
        k2 = np.array(equations(t[i] + dt / 2, z[:, i] + dt / 2 * k1))  # Second slope
        k3 = np.array(equations(t[i] + dt / 2, z[:, i] + dt / 2 * k2))  # Third slope
        k4 = np.array(equations(t[i] + dt, z[:, i] + dt * k3))  # Fourth slope
        z[:, i + 1] = z[:, i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # RK4 update

    return t, z

# Number of steps for RK4 method
n_steps_rk4 = 6000
# Solve the system using RK4 method
t_rk4, z_rk4 = rk4_method(equations, [0, t_final], z0, n_steps_rk4)

# Plot the orbit calculated by RK4 method
plt.figure(figsize=(8, 6))
plt.plot(z_rk4[0], z_rk4[2], label='Orbit (RK4)')  # Plot position in the y1-y2 plane
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object (RK4 Method)')
plt.grid()  # Add grid for better visualization
plt.legend()  # Add legend to the plot
plt.show()
plt.savefig('./pset3_RK4.png')  # Save the plot to a file

# Print the number of steps used by solve_ivp
print("Number of steps used by solve_ivp:", len(solution.t))

# Extend the simulation to cover three orbital periods
t_final_3 = 3 * t_final  # New final time for three periods
solution_3 = solve_ivp(equations, [0, t_final_3], z0, method='RK45', rtol=1e-12, atol=1e-12)

# Plot the orbit over three periods
plt.figure(figsize=(8, 6))
plt.plot(solution_3.y[0], solution_3.y[2], label='Orbit (3 Periods)')  # Plot position in the y1-y2 plane
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Orbit of the Object over 3 Periods (solve_ivp)')
plt.grid()  # Add grid for better visualization
plt.legend()  # Add legend to the plot
plt.show()
plt.savefig('./pset3_3_periods.png')  # Save the plot to a file