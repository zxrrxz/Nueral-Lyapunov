import numpy as np
from scipy.linalg import eig

# 系统参数
Xg = 0.5
f_base = 50.0          # 基频，用于计算 Lg
omega_base = 100 * np.pi
Lg = Xg / omega_base    # 0.5 / (100π) ≈ 0.0015915
kp_pll = 10 * np.pi     # ≈ 31.4159
ki_pll = 100 * kp_pll   # ≈ 3141.59
ug = 1.0
id_ = 1.0

# 常数 a
a = 1 - kp_pll * Lg * id_
print(f"a = {a:.6f}")  # 应不为零

# 平衡点 delta_eq 的两个解
delta_eq_stable = np.arcsin(Xg * id_ / ug)        # π/6 ≈ 0.5236
delta_eq_unstable = np.pi - delta_eq_stable       # 5π/6 ≈ 2.6180
x_eq = 0.0   # 对于两个平衡点均为 0

print(f"稳定平衡点 delta_eq = {delta_eq_stable:.6f}")
print(f"不稳定平衡点 delta_eq = {delta_eq_unstable:.6f}")

# 选择不稳定平衡点进行分析
delta_eq = delta_eq_unstable
cos_delta = np.cos(delta_eq)   # cos(5π/6) = -√3/2 ≈ -0.8660254

# 构建雅可比矩阵 A
A = (1/a) * np.array([
    [-kp_pll * ug * cos_delta, 1],
    [-ki_pll * ug * cos_delta, ki_pll * Lg * id_]
])

print("\n雅可比矩阵 A (在不稳定平衡点):\n", A)

# 特征分解
eigvals, eigvecs = eig(A)
print("\n特征值:\n", eigvals)

# 区分稳定/不稳定特征向量（实部<0为稳定）
stable_idx = np.real(eigvals) < 0
unstable_idx = np.real(eigvals) > 0

v_stable = eigvecs[:, stable_idx]
v_unstable = eigvecs[:, unstable_idx]

if v_stable.size > 0:
    # 若有多列，通常取第一个（本系统二维，且鞍点应只有一个稳定特征值）
    v = v_stable[:, 0]
    v = v / np.linalg.norm(v)   # 归一化
    print("\n稳定特征向量 (归一化):\n", v)
else:
    print("没有稳定特征向量")