import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(u, t, sigma, rho, beta):
    x, y, z = u
    dx_dt = sigma * (y - x)
    dy_dt = rho * x - y - x * z
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def rk2_step(u, t, h, f, sigma, rho, beta):
    # Midpoint method
    k1 = f(u, t, sigma, rho, beta)
    u_tilde = u + 0.5 * h * k1
    k2 = f(u_tilde, t + 0.5 * h, sigma, rho, beta)
    u_next = u + h * k2
    return u_next


def simulate_lorenz(u0, sigma, rho, beta, h, T):
    t_array = np.arange(0, T + h, h)
    u = np.zeros((len(t_array), 3))
    u[0] = u0
    for i in range(len(t_array) - 1):
        u[i + 1] = rk2_step(u[i], t_array[i], h, lorenz, sigma, rho, beta)
    return t_array, u


# 예: 여러 초기조건에 대해 하나의 파라미터 셋으로 실험
def experiment_multiple_initial_conditions(sigma, rho, beta, initial_conditions, h=0.01, T=50):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for ic in initial_conditions:
        t_array, sol = simulate_lorenz(ic, sigma, rho, beta, h, T)
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label=f'IC={ic}')
    ax.set_title(f'Lorenz System (σ={sigma}, ρ={rho}, β={beta})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# 예: 하나의 초기조건에 대해 여러 파라미터 조합 실험
def experiment_multiple_parameters(u0, param_sets, h=0.01, T=50):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for (sigma, rho, beta) in param_sets:
        t_array, sol = simulate_lorenz(u0, sigma, rho, beta, h, T)
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label=f'σ={sigma}, ρ={rho}, β={beta}')
    ax.set_title('Lorenz System with Multiple Parameter Sets')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# 파라미터 스윕 예시: rho 값을 다양한 값으로 바꿔가며 실험
def experiment_parameter_sweep(u0, sigma_list, rho_list, beta_list, h=0.01, T=50):
    # sigma_list, rho_list, beta_list 중 하나만 길이가 1보다 클 경우 여러 개 파라미터를 실험할 수 있음.
    # 여기서는 예를 들어 rho 값만 바꿔가며 실험한다고 가정.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for sigma in sigma_list:
        for rho in rho_list:
            for beta in beta_list:
                t_array, sol = simulate_lorenz(u0, sigma, rho, beta, h, T)
                ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label=f'σ={sigma}, ρ={rho}, β={beta}')

    ax.set_title('Lorenz System Parameter Sweep')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# ---------------------
# 여기서 실제 실험 예시를 구성해보자.
# ---------------------

if __name__ == "__main__":
    # 실험 1: 특정 파라미터(σ=10,ρ=28,β=8/3)에서 여러 초기조건 실험
    initial_conditions_list = [
        [1.0, 1.0, 1.0],
        [1.01, 1.0, 1.0],
        [-5.0, 5.0, 20.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 2.0]
    ]
    experiment_multiple_initial_conditions(10.0, 28.0, 8.0 / 3.0, initial_conditions_list, h=0.01, T=50)

    # 실험 2: 하나의 초기조건(1,1,1)에서 파라미터 셋 변경
    param_sets = [
        (10.0, 10.0, 8.0 / 3.0),
        (10.0, 22.0, 8.0 / 3.0),
        (10.0, 28.0, 8.0 / 3.0),
        (10.0, 35.0, 8.0 / 3.0)  # 더 큰 rho로 실험
    ]
    experiment_multiple_parameters([1.0, 1.0, 1.0], param_sets, h=0.01, T=50)

    # 실험 3: 파라미터 스윕
    # rho 값들을 10, 20, 28, 35로 변화시키며 하나의 σ=10, β=8/3에 대해 실험
    sigma_list = [10.0]
    rho_list = [10.0, 20.0, 28.0, 35.0]
    beta_list = [8.0 / 3.0]
    experiment_parameter_sweep([1.0, 1.0, 1.0], sigma_list, rho_list, beta_list, h=0.01, T=50)

    # 실험 4: 더 작은 스텝사이즈로 장시간 시뮬레이션 (안정성과 정확성 비교)
    # 초기조건 (1,1,1), σ=10, ρ=28, β=8/3 에서 h=0.001, T=100으로
    t_array, sol = simulate_lorenz([1.0, 1.0, 1.0], 10.0, 28.0, 8.0 / 3.0, 0.001, 100)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], color='blue')
    ax.set_title('Lorenz Attractor (Long Simulation with smaller step h=0.001, T=100)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()