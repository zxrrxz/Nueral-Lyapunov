import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# GFL
Xg = 0.5
Lg = Xg / (100 * np.pi)          # ≈ 0.0015915
kp_pll = 10 * np.pi              # ≈ 31.4159
ki_pll = 100 * kp_pll            # ≈ 3141.59
ug = 1.0
id = 1.0
delta_range = np.pi
x_int_range = 75

# ---------- 2. 计算平衡点 ----------
delta_eq = np.arcsin(Xg * id / ug)        # π/6 ≈ 0.5236
x_eq = 0
delta_uep = np.pi - 2*delta_eq
x_uep = 0

# ------------------------------
# 1. 定义目标系统：单机无穷大 GFM，二阶摇摆方程
# ------------------------------
def f(x):
    # GFL
    delta_input = x[:, 0]
    x_input = x[:, 1]

    delta = delta_input
    x = x_input

    a = 1 - kp_pll * Lg * id  # 常数
    b = Xg * id - ug * torch.sin(delta+delta_eq)

    dx = (ki_pll * b + ki_pll * Lg * id * x) / a
    ddelta = (kp_pll * b + x) / a

    return torch.stack([ddelta, dx], dim=1)


# ------------------------------
# 2. 生成数据集：采样状态空间中的点，计算对应的时间导数
# ------------------------------
def generate_dataset(n_samples=3500, x_range=[-delta_range, delta_range], y_range=[-x_int_range, x_int_range],
                     center_samples=2000, center_radius=0.05):
    delta_samples = np.random.uniform(x_range[0], x_range[1], n_samples)
    omega_samples = np.random.uniform(y_range[0], y_range[1], n_samples)
    x = np.stack([delta_samples, omega_samples], axis=1)
    delta_center = np.random.uniform(-np.pi*center_radius, np.pi*center_radius, center_samples)
    x_center = np.random.uniform(-75*center_radius, 75*center_radius, center_samples)
    x_center_orig = np.stack([delta_center, x_center], axis=1)

    x_orig = np.concatenate([x, x_center_orig], axis=0)
    x_tensor = torch.tensor(x_orig, dtype=torch.float32)
    with torch.no_grad():
        dx_tensor = f(x_tensor)
    return x_tensor, dx_tensor

# ------------------------------
# 3. 定义神经网络：用于近似 Lyapunov 函数 V(delta, omega)
# ------------------------------
class LyapunovNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        V_net = self.net(x)  # 形状 (batch, 1)
        # 强制 V(0) = 0：减去原点处的预测值
        # if not hasattr(self, 'V0') or self.V0 is None:
        #     with torch.no_grad():
        #         x0 = torch.zeros(1, x.shape[1], device=x.device)
        #         self.V0 = self.net(x0)  # 标量，shape (1,1)
        return V_net #- self.V0


def compute_lie_derivative(model, x):
    """计算 Lie 导数 dV/dt = grad(V) * f(x)"""
    x.requires_grad_(True)
    V = model(x).squeeze()
    grad_V = torch.autograd.grad(V.sum(), x, create_graph=True)[0]  # shape (batch, 2)
    dx = f(x)
    lie = (grad_V * dx).sum(dim=1)
    return lie


def lyapunov_loss(model, x, dx, alpha, beta, gamma, a, b, c):
    V = model(x).squeeze()

    lie = compute_lie_derivative(model, x)

    # 条件1: V(x) > 0 (对于非零点)
    pos_def = torch.relu(-V)

    # 条件2: dV/dt < 0 (对于非零点)
    neg_def = torch.relu(lie)  # 允许小容差 tau

    # 条件3: V(0) = 0
    x0 = torch.tensor([[0, 0]], dtype=x.dtype, device=x.device)
    V0 = model(x0).squeeze()
    zero_penalty = V0 ** 2

    roa_term = (b*torch.norm(x, dim=1) ** 2 - c * V).mean()

    loss = alpha * pos_def.mean() + beta * neg_def.mean() + gamma * zero_penalty + a*roa_term
    return loss

def verify(V_net, x1_range=(-delta_range, delta_range), x2_range=(-x_int_range, x_int_range), grid_size=50, eps=1e-5):
    """在网格上验证 Lyapunov 条件，返回 (是否有效, 违反点列表)"""
    x1 = torch.linspace(x1_range[0], x1_range[1], grid_size)
    x2 = torch.linspace(x2_range[0], x2_range[1], grid_size)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_vals = torch.stack([X1.ravel(), X2.ravel()], dim=1)  # (N,2)

    # 计算 V 值（无梯度）
    with torch.no_grad():
        V_vals = V_net(x_vals).numpy()

    # 计算李导数（需要梯度）
    x_grad = x_vals.clone().detach().requires_grad_(True)
    V_grad = V_net(x_grad)
    gradV = torch.autograd.grad(V_grad.sum(), x_grad, create_graph=False)[0]
    fx = f(x_grad)  # 调用之前定义的动力学函数
    lie = (gradV * fx).sum(dim=1).detach().numpy()

    # 忽略原点附近点
    x_np = x_vals.numpy()
    mask = (np.abs(x_np[:, 0]) > eps) | (np.abs(x_np[:, 1]) > eps)

    violation_pos = (V_vals[mask] <= 0)
    violation_der = (lie[mask] >= 0)
    if np.any(violation_pos) or np.any(violation_der):
        # 收集所有违反点
        x_viol = x_vals[mask][violation_pos | violation_der]
        return False, x_viol
    else:
        return True, None

# ------------------------------
# 5. 训练过程
# ------------------------------
def train_lyapunov(model, optimizer, n_epochs=2000, batch_size=512,
                   alpha=1.0, beta=1.0, gamma=1.0, a=1.0,b=1.0,c=0.01):
    x_train, dx_train = generate_dataset()
    dataset = torch.utils.data.TensorDataset(x_train, dx_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, batch_dx in dataloader:
            optimizer.zero_grad()
            loss = lyapunov_loss(model, batch_x, batch_dx, alpha, beta, gamma, a,b,c)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # model._ensure_U_nonnegative()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss_history[-1]:.4f}")
    return loss_history


def plot_lyapunov_3d(model, x_range=(-delta_range, delta_range), y_range=(-x_int_range, x_int_range), n_points=500):
    delta_vals = np.linspace(x_range[0], x_range[1], n_points)
    omega_vals = np.linspace(y_range[0], y_range[1], n_points)
    Delta, Omega = np.meshgrid(delta_vals, omega_vals)
    grid = np.stack([Delta.ravel(), Omega.ravel()], axis=1)
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        V_vals = model(grid_tensor).squeeze().numpy()
    V_vals = V_vals.reshape(Delta.shape)

    lie = compute_lie_derivative(model, grid_tensor).detach().numpy()
    lie = lie.reshape(Delta.shape)

    fig = plt.figure(figsize=(14, 6))
    # 左子图：V 曲面 + 底面等高线
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Delta, Omega, V_vals, cmap='RdBu_r', edgecolor='none', alpha=0.8)
    z_min = V_vals.min()
    ax1.contour(Delta, Omega, V_vals, levels=12, zdir='z', offset=z_min,
                cmap='RdBu_r', alpha=0.8, linewidths=1)
    ax1.set_xlabel('δ')
    ax1.set_ylabel('x')
    ax1.set_zlabel('V')
    ax1.set_title('Learned Lyapunov Function V')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 右子图：dV/dt
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Delta, Omega, lie, cmap='viridis', edgecolor='none', alpha=0.7)
    ax2.set_xlabel('δ')
    ax2.set_ylabel('x')
    ax2.set_zlabel('dV/dt')
    ax2.set_title('Time Derivative')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()


def integrate_trajectory(x0, t_span=(0, 10), t_eval=None):
    def ode_func(t, x):
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        dx = f(x_tensor).detach().numpy().flatten()
        return dx

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 3000)
    sol = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, method='RK45')
    return sol.y.T  # shape (N,2)


def backward(t_span=(0, 10), t_eval=None):
    def ode_func(t, x):
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        dx = f(x_tensor).detach().numpy().flatten()
        return -dx

    # x0_pos = [2.888092-0.999561*0.001, 0.025001+0.029644]
    # x0_neg = [2.888092+0.999561*0.001, 0.025001-0.029644]
    x0_pos = [delta_uep - 0.9995794 * 0.001, x_uep + 0.02901282 * 0.001]
    x0_neg = [delta_uep + 0.9995794 * 0.001, x_uep - 0.02901282 * 0.001]
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 3000)
    sol_pos = solve_ivp(ode_func, t_span, x0_pos, t_eval=t_eval, method='RK45')
    sol_neg = solve_ivp(ode_func, t_span, x0_neg, t_eval=t_eval, method='RK45')

    traj_pos = sol_pos.y.T
    traj_neg = sol_neg.y.T
    traj_all = np.vstack([traj_neg[::-1], traj_pos])
    return traj_all  # shape (N,2)


def plot_ROA(V_net, d_star, a=delta_range, b=x_int_range,
             x_range=(-delta_range, delta_range), y_range=(-x_int_range, x_int_range),
             resolution=500):
    delta_vals = np.linspace(x_range[0], x_range[1], resolution)
    omega_vals = np.linspace(y_range[0], y_range[1], resolution)
    Delta, Omega = np.meshgrid(delta_vals, omega_vals)
    points = torch.tensor(np.stack([Delta.ravel(), Omega.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        V_grid = V_net(points).numpy().reshape(Delta.shape)

    plt.figure(figsize=(10, 6))
    # 颜色图（整个矩形区域）
    im = plt.pcolormesh(Delta, Omega, V_grid, shading='auto', cmap='RdBu_r')
    plt.colorbar(im, label='V(δ, ω)')

    # 稳定域边界（红色等高线）
    contour = plt.contour(Delta, Omega, V_grid, levels=[d_star], colors='red', linewidths=2, linestyles='-')
    plt.clabel(contour, inline=True, fontsize=10, fmt=f'V = {d_star:.4f}')
    # d1 = 1.5*d_star
    # contour = plt.contour(Delta, Omega, V_grid, levels=[d1], colors='red', linewidths=2, linestyles='-')
    # plt.clabel(contour, inline=True, fontsize=10, fmt=f'V = {d1:.4f}')
    # 采样椭圆边界（黑色虚线）
    # theta = np.linspace(0, 2*np.pi, 200)
    # delta_ellipse = a * np.cos(theta)
    # omega_ellipse = b * np.sin(theta)
    # plt.plot(delta_ellipse, omega_ellipse, 'k--', linewidth=1.5, label='Sampling ellipse')

    # 平衡点
    plt.plot(0, 0, 'ko', markersize=6, label='Equilibrium')
    plt.plot(delta_uep, 0, 'ko', markersize=6, label='uep')

    # initial_point = [-2.5, -0.1]
    # traj = integrate_trajectory(initial_point)
    # plt.plot(traj[:, 0], traj[:, 1], 'black', linewidth=1.5, alpha=0.8, label='Trajectory')
    # plt.plot(initial_point[0], initial_point[1], 'black', markersize=8, label='Start point')

    traj1 = backward()
    plt.plot(traj1[:, 0], traj1[:, 1], 'k-', lw=1.5)

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel('δ (rad)')
    plt.ylabel('x p.u.')
    plt.title(f'Stability Region (V < {d_star:.6f}) and Sampling Ellipse')
    plt.grid(alpha=0.3)
    # plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))
    plt.show()


def compute_V_at_point(V_net, delta, omega):
    """
    计算给定点 (delta, omega) 处的 Lyapunov 函数值
    """
    # 构造输入张量，形状 (1, 2)
    x = torch.tensor([[delta, omega]], dtype=torch.float32)
    with torch.no_grad():  # 不需要梯度，提高效率
        V = V_net(x)
    return V.item()  # 返回 Python 浮点数


# ------------------------------
# 7. 主程序
# ------------------------------
if __name__ == "__main__":
    # 初始化模型和优化器
    model = LyapunovNN(hidden_dim=32)
    # model = ICNN(hidden_dims=[16, 16])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练
    print("开始训练神经 Lyapunov 函数...")
    loss_hist = train_lyapunov(model, optimizer, n_epochs=910,
                               alpha=0.9, beta=0.8, gamma=2.4, a=1.0 , b=0.01 , c=0.01)
    # plt.plot(loss_hist)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()

    # 可视化
    # plot_lyapunov_3d(model)
    delta_val = delta_uep
    x_val = x_eq
    d_star = 0.9766*compute_V_at_point(model, delta_val, x_val)
    plot_ROA(model, d_star, a=delta_range, b=x_int_range)