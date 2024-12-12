import numpy as np

# 함수 정의
def f(x, y):
    return np.exp(x * y)

# Simpson's 1D method
def simpson_1d(f_vals, h):
    n = len(f_vals)
    if n % 2 == 0:
        raise ValueError("Simpson의 방법은 홀수 개의 점이 필요합니다.")
    return (h / 3) * (f_vals[0] + f_vals[-1] + 4 * np.sum(f_vals[1:-1:2]) + 2 * np.sum(f_vals[2:-2:2]))

# 이중적분 함수
def double_integral_rectangular(f, x_vals, y_vals):
    hx = x_vals[1] - x_vals[0]  # x 방향 간격
    hy = y_vals[1] - y_vals[0]  # y 방향 간격

    # 각 y에 대해 x 방향 적분
    Ix_for_each_y = []
    for y in y_vals:
        f_x = f(x_vals, y)  # f(x, y)의 값을 계산
        Ix_for_each_y.append(simpson_1d(f_x, hx))  # x 방향 적분 결과 저장

    # y 방향 적분
    return simpson_1d(Ix_for_each_y, hy)

# 적분 범위 설정
N = 101  # Simpson's 방법에서는 홀수여야 함
x_vals = np.linspace(0, 1, N)
y_vals = np.linspace(0, 1, N)

# 이중적분 계산
result_a = double_integral_rectangular(f, x_vals, y_vals)
print("사각형 영역에서의 이중적분 결과:", result_a)

# 극좌표 변환된 함수 정의
def g(r, theta):
    return np.exp(r**2 * np.cos(theta) * np.sin(theta)) * r

# 이중적분 함수 (극좌표)
def double_integral_polar(g, r_vals, theta_vals):
    hr = r_vals[1] - r_vals[0]  # r 방향 간격
    hth = theta_vals[1] - theta_vals[0]  # theta 방향 간격

    # 각 theta에 대해 r 방향 적분
    Ir_for_each_theta = []
    for theta in theta_vals:
        f_r = g(r_vals, theta)  # f(r, theta) 계산
        Ir_for_each_theta.append(simpson_1d(f_r, hr))  # r 방향 적분 결과 저장

    # theta 방향 적분
    return simpson_1d(Ir_for_each_theta, hth)

# 적분 범위 설정
N = 101  # Simpson's 방법에서는 홀수여야 함
r_vals = np.linspace(0, 1, N)
theta_vals = np.linspace(0, np.pi / 2, N)

# 이중적분 계산
result_b = double_integral_polar(g, r_vals, theta_vals)
print("1사분면 원 영역에서의 이중적분 결과:", result_b)
