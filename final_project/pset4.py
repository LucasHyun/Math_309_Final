import numpy as np
import matplotlib.pyplot as plt

# ===== PDE 정의 및 수치방법 설정 =====
# PDE: u_t + u_x = 0
# BC: periodic
# IC: u(0,x) = sin(x)
L = 2*np.pi
N = 100   # (a) x방향 100등분
dx = L/N
x = np.linspace(0, L, N, endpoint=False)
u0 = np.sin(x)

def spatial_derivative(u, dx):
    return (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)

def f(u, t, dx):
    return -spatial_derivative(u, dx)

def rk4_step(u, t, dt, dx):
    k1 = f(u, t, dx)
    k2 = f(u + 0.5*dt*k1, t + 0.5*dt, dx)
    k3 = f(u + 0.5*dt*k2, t + 0.5*dt, dx)
    k4 = f(u + dt*k3, t + dt, dx)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve(u_initial, dt, T, dx):
    steps = int(T/dt)
    u = u_initial.copy()
    t = 0.0
    for _ in range(steps):
        u = rk4_step(u, t, dt, dx)
        t += dt
    return u, t

# ===== (c) dt 변화시키기 =====
dts = [0.1, 0.05, 0.01, dx, dx/2, dx/4, dx/8, dx/20]
T_values = [1.0, 10.0]

solutions = {}
for T in T_values:
    solutions[T] = {}
    for dt_test in dts:
        u_approx, _ = solve(u0, dt_test, T, dx)
        solutions[T][dt_test] = u_approx

# ===== (d) T=1, T=10에서의 결과 plotting =====
plt.figure(figsize=(10,6))
for i, T in enumerate(T_values, start=1):
    plt.subplot(2,1,i)
    for dt_test in dts:
        plt.plot(x, solutions[T][dt_test], label=f"dt={dt_test:.5f}")
    plt.title(f"u(x) at t={T}")
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
plt.tight_layout()
plt.show()

# ===== (e) 참조해 계산 및 에러 분석 =====
# 참조해: dt_ref = dx/20 으로 매우 작은 dt 사용
dt_ref = dx/20
T_ref = 1.0
u_ref, _ = solve(u0, dt_ref, T_ref, dx)

# 다양한 dt에 대해 해를 구하고 참조해와 비교 -> 에러 계산
test_dts = [dx, dx/2, dx/4, dx/8, dx/16]

def l_inf_error(u_test, u_ref):
    return np.max(np.abs(u_test - u_ref))

def l2_error(u_test, u_ref, dx):
    diff = u_test - u_ref
    return np.sqrt(np.sum(diff**2)*dx)

e_inf_list = []
e_l2_list = []
for dt_test in test_dts:
    u_test, _ = solve(u0, dt_test, T_ref, dx)
    e_inf = l_inf_error(u_test, u_ref)
    e_l2 = l2_error(u_test, u_ref, dx)
    e_inf_list.append(e_inf)
    e_l2_list.append(e_l2)

# 에러 vs dt를 log-log 플롯
plt.figure()
plt.loglog(test_dts, e_inf_list, 'o-', label='L_inf error')
plt.loglog(test_dts, e_l2_list, 's-', label='L2 error')
plt.xlabel('dt')
plt.ylabel('Error')
plt.title('Error vs dt (log-log)')
plt.grid(True)
plt.legend()
plt.show()

# 에러 비율 및 log2(e1/e2) 계산
# 예를 들어 dt를 순서대로 반으로 줄일 때의 에러비 확인
# dt: dx, dx/2, dx/4, dx/8, dx/16
# e_inf_list, e_l2_list도 같은 순서
# log2(e(i)/e(i+1)) 계산 (인접 dt 비율)
def log2_ratio(e1, e2):
    return np.log2(e1/e2)

print("L_inf error ratios in log2 scale:")
for i in range(len(e_inf_list)-1):
    ratio = log2_ratio(e_inf_list[i], e_inf_list[i+1])
    print(f"log2(e_inf(dt={test_dts[i]:.5f}) / e_inf(dt={test_dts[i+1]:.5f})) = {ratio:.4f}")

print("\nL2 error ratios in log2 scale:")
for i in range(len(e_l2_list)-1):
    ratio = log2_ratio(e_l2_list[i], e_l2_list[i+1])
    print(f"log2(e_l2(dt={test_dts[i]:.5f}) / e_l2(dt={test_dts[i+1]:.5f})) = {ratio:.4f}")