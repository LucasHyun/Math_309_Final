import numpy as np

# Function definition
def f(x, y):
    return np.exp(x * y)

# Simpson's 1D method
def simpson_1d(f_vals, h):
    n = len(f_vals)
    if n % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    return (h / 3) * (f_vals[0] + f_vals[-1] + 4 * np.sum(f_vals[1:-1:2]) + 2 * np.sum(f_vals[2:-2:2]))

# Double integral function (rectangular coordinates)
def double_integral_rectangular(f, x_vals, y_vals):
    hx = x_vals[1] - x_vals[0]  # Step size in the x-direction
    hy = y_vals[1] - y_vals[0]  # Step size in the y-direction

    # Perform integration in the x-direction for each y
    Ix_for_each_y = []
    for y in y_vals:
        f_x = f(x_vals, y)  # Compute f(x, y) values
        Ix_for_each_y.append(simpson_1d(f_x, hx))  # Store the result of integration in the x-direction

    # Perform integration in the y-direction
    return simpson_1d(Ix_for_each_y, hy)

# Set integration range
N = 101  # Simpson's rule requires an odd number of points
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)

# Compute double integral
result_a = double_integral_rectangular(f, x_vals, y_vals)
print("Double integral result over a rectangular region:", result_a)

# Function definition in polar coordinates
def g(r, theta):
    return np.exp(r**2 * np.cos(theta) * np.sin(theta)) * r

# Double integral function (polar coordinates)
def double_integral_polar(g, r_vals, theta_vals):
    hr = r_vals[1] - r_vals[0]  # Step size in the r-direction
    hth = theta_vals[1] - theta_vals[0]  # Step size in the theta-direction

    # Perform integration in the r-direction for each theta
    Ir_for_each_theta = []
    for theta in theta_vals:
        f_r = g(r_vals, theta)  # Compute f(r, theta) values
        Ir_for_each_theta.append(simpson_1d(f_r, hr))  # Store the result of integration in the r-direction

    # Perform integration in the theta-direction
    return simpson_1d(Ir_for_each_theta, hth)

# Set integration range
N = 101  # Simpson's rule requires an odd number of points
r_vals = np.linspace(0, 1, N)
theta_vals = np.linspace(0, np.pi / 2, N)

# Compute double integral
result_b = double_integral_polar(g, r_vals, theta_vals)
print("Double integral result over the first quadrant of a circular region:", result_b)