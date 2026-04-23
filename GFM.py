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

#GFM
M = 2  # 惯性常数
D = 10  # 阻尼系数
Pm = 1
Pe = 2
delta_s = torch.arcsin(torch.tensor(Pm / Pe))
delta_uep = np.pi-2*delta_s
delta_range = np.pi
omega_range = 0.2
# ------------------------------
# 1. 定义目标系统：单机无穷大 GFM，二阶摇摆方程
# ------------------------------
def f(x):

    delta, omega = x[:, 0], x[:, 1]
    ddelta = omega * 100 * np.pi
    domega = (Pm - Pe * torch.sin(delta + delta_s) - D * omega) / M

    return torch.stack([ddelta, domega], dim=1)

    #GFL

# ------------------------------
# 2. 生成数据集：采样状态空间中的点，计算对应的时间导数
# ------------------------------
def generate_dataset(n_samples=1000, x_range=[-delta_range, delta_range], y_range=[-omega_range, omega_range]):
    delta_samples = np.random.uniform(x_range[0] , x_range[1], n_samples)
    omega_samples = np.random.uniform(y_range[0], y_range[1], n_samples)
    x = np.stack([delta_samples, omega_samples], axis=1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
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

class ICNN(nn.Module):
    """
    输入凸神经网络 (Input Convex Neural Network)
    用于参数化 Lyapunov 函数 V(x)，保证 V 关于 x 凸，且 V(0)=0
    """
    def __init__(self, input_dim=2, hidden_dims=[16, 16], activation='softplus'): #softplus tanh
        super(ICNN, self).__init__()
        self.activation = activation
        self.num_layers = len(hidden_dims)

        # 定义各层的权重和偏置
        # W_i: 线性部分（与输入直连）形状 (hidden_i, input_dim)
        # U_i: 凸性保证部分（非负）形状 (hidden_i, hidden_{i-1})
        self.W = nn.ModuleList()
        self.U = nn.ModuleList()
        self.b = nn.ParameterList()

        # 第一层特殊处理：z1 = activation(W0 * x + b0)
        self.W0 = nn.Linear(input_dim, hidden_dims[0], bias=True)
        self.U0 = None  # 第一层没有U

        # 后续层
        prev_dim = hidden_dims[0]
        for i, hdim in enumerate(hidden_dims[1:], start=1):
            # 线性部分
            self.W.append(nn.Linear(input_dim, hdim, bias=False))  # 偏置单独处理
            # 凸性部分（非负权重）
            self.U.append(nn.Linear(prev_dim, hdim, bias=False))
            # 偏置项
            self.b.append(nn.Parameter(torch.zeros(hdim)))
            prev_dim = hdim

        # 输出层：将最后一层映射到标量
        self.output_layer = nn.Linear(prev_dim, 1, bias=False)

        # 初始化权重（使用适当的初始化）
        self._init_weights()

    def _init_weights(self):
        # 对 W0 和 W 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.W0.weight)
        nn.init.zeros_(self.W0.bias)
        for layer in self.W:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.U:
            nn.init.xavier_uniform_(layer.weight)
            # 强制初始 U 非负（可以取绝对值）
            with torch.no_grad():
                layer.weight.data.abs_()
        nn.init.xavier_uniform_(self.output_layer.weight)

    def _ensure_U_nonnegative(self):
        """在优化器步后调用，确保 U 矩阵元素非负"""
        for layer in self.U:
            layer.weight.data.clamp_(min=0)

    def forward(self, x):
        """
        前向传播计算 V(x)
        """
        # 第一层
        z = self.W0(x)                     # (batch, hidden0)
        z = self._activation(z)

        # 中间层
        for i in range(self.num_layers - 1):
            w_part = self.W[i](x)          # (batch, hidden_{i+1})
            u_part = self.U[i](z)          # (batch, hidden_{i+1})
            z = u_part + w_part + self.b[i]  # 加上偏置
            z = self._activation(z)

        # 输出层
        v = self.output_layer(z)            # (batch, 1)

        # 保证 V(0)=0：减去 V(0) 的估计值
        # 此处通过构造一个零输入计算 V(0)
        if not hasattr(self, '_V0'):
            # 计算 V(0) 并缓存
            zero_input = torch.zeros_like(x[:1])
            with torch.no_grad():
                self._V0 = self.forward_no_grad(zero_input)
        return v - self._V0

    def forward_no_grad(self, x):
        """用于计算 V(0) 的副本，不带梯度"""
        z = self.W0(x)
        z = self._activation(z)
        for i in range(self.num_layers - 1):
            w_part = self.W[i](x)
            u_part = self.U[i](z)
            z = u_part + w_part + self.b[i]
            z = self._activation(z)
        v = self.output_layer(z)
        return v

    def _activation(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'softplus':
            return F.softplus(x)
        else:
            raise ValueError("Unsupported activation")

def compute_lie_derivative(model, x):
    """计算 Lie 导数 dV/dt = grad(V) * f(x)"""
    x.requires_grad_(True)
    V = model(x).squeeze()
    grad_V = torch.autograd.grad(V.sum(), x, create_graph=True)[0]  # shape (batch, 2)
    dx = f(x)
    lie = (grad_V * dx).sum(dim=1)
    return lie

# ------------------------------
# 4. 损失函数：基于论文中的 Lyapunov 风险
#    条件： V(0)=0, V(x)>0 (x!=0), dV/dt < 0 (x!=0)
# ------------------------------
def lyapunov_loss(model, x, dx, alpha, beta, gamma, c):
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
    roa_term = (0.1*torch.norm(x, dim=1) ** 2 - c * V).mean() #
    # 综合损失
    loss = alpha * pos_def.mean() + beta * neg_def.mean()  + gamma * zero_penalty + roa_term
    return loss

# ------------------------------
# 5. 训练过程
# ------------------------------
def train_lyapunov(model, optimizer, n_epochs=2000, batch_size=512,
                   alpha=1.0, beta=1.0, gamma=1.0,c=0.01):
    x_train, dx_train = generate_dataset(n_samples=2000)
    dataset = torch.utils.data.TensorDataset(x_train, dx_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, batch_dx in dataloader:
            optimizer.zero_grad()
            loss = lyapunov_loss(model, batch_x, batch_dx, alpha, beta, gamma,c)
            loss.backward()
            optimizer.step()
            # model._ensure_U_nonnegative()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss_history[-1]:.4f}")
    return loss_history

def plot_lyapunov_3d(model, x_range=(-delta_range, delta_range), y_range=(-omega_range, omega_range), n_points=200):
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
    ax1.set_ylabel('ω')
    ax1.set_zlabel('V')
    ax1.set_title('Learned Lyapunov Function V')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 右子图：dV/dt
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Delta, Omega, lie, cmap='viridis', edgecolor='none', alpha=0.7)
    ax2.set_xlabel('δ')
    ax2.set_ylabel('ω')
    ax2.set_zlabel('dV/dt')
    ax2.set_title('Time Derivative')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

def estimate_d_star_on_ellipse(V_net, a=delta_range, b=omega_range, n_samples=500):
    """
    在椭圆边界 (δ²/a² + ω²/b² = 1) 上均匀采样，估计 d* = min V(x)
    """
    angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    delta_vals = a * np.cos(angles)
    omega_vals = b * np.sin(angles)
    points = torch.tensor(np.column_stack([delta_vals, omega_vals]), dtype=torch.float32)

    with torch.no_grad():
        V_vals = V_net(points).numpy()

    d_star = np.min(V_vals)
    print(f"Estimated d* = {d_star:.6f}")
    return d_star, points.numpy()

def integrate_trajectory(x0, t_span=(0, 10), t_eval=None):
    def ode_func(t, x):
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        dx = f(x_tensor).detach().numpy().flatten()
        return dx

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, method='RK45')
    return sol.y.T  # shape (N,2)

def backward(t_span=(0, 10), t_eval=None):

    def ode_func(t, x):
        x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        dx = f(x_tensor).detach().numpy().flatten()
        return -dx

    x0_pos = [2.0933965692132, 0.00005413805]
    x0_neg = [2.0953936355732, -0.00005413805]
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol_pos = solve_ivp(ode_func, t_span, x0_pos, t_eval=t_eval, method='RK45')
    sol_neg = solve_ivp(ode_func, t_span, x0_neg, t_eval=t_eval, method='RK45')

    traj_pos = sol_pos.y.T
    traj_neg = sol_neg.y.T
    traj_all = np.vstack([traj_neg[::-1], traj_pos])
    return traj_all  # shape (N,2)

def plot_ROA(V_net, d_star,a=delta_range, b=omega_range,
                                       x_range=(-delta_range, delta_range), y_range=(-omega_range, omega_range),
                                       resolution=300):
    """
    绘制整个矩形区域的 Lyapunov 函数颜色图，
    红色等高线为稳定域边界 V = d*，
    黑色虚线为采样椭圆边界。
    """
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
    plt.ylabel('ω p.u.')
    plt.title(f'Real and estimated Stability Region (V < {d_star:.6f})')
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
    with torch.no_grad():   # 不需要梯度，提高效率
        V = V_net(x)
    return V.item()   # 返回 Python 浮点数

# def plot_Grad(V_net,a=delta_range, b=omega_range,
#                                        x_range=(-delta_range, delta_range), y_range=(-omega_range, omega_range),
#                                        resolution=300):
#     delta_vals = np.linspace(x_range[0], x_range[1], resolution)
#     omega_vals = np.linspace(y_range[0], y_range[1], resolution)
#     Delta, Omega = np.meshgrid(delta_vals, omega_vals)
#     points = torch.tensor(np.stack([Delta.ravel(), Omega.ravel()], axis=1), dtype=torch.float32)
#
#     with torch.no_grad():
#         V_grid = V_net(points).numpy().reshape(Delta.shape)
#     lie = compute_lie_derivative(model, points).detach().numpy()
#     lie = lie.reshape(Delta.shape)
#     plt.figure(figsize=(10, 6))
#     # 颜色图（整个矩形区域）
#     im = plt.pcolormesh(Delta, Omega, lie, shading='auto', cmap='RdBu_r')
#     plt.colorbar(im, label='V(δ, ω)')
#
#     # 稳定域边界（红色等高线）
#     contour = plt.contour(Delta, Omega, V_grid, levels=[d_star], colors='red', linewidths=2, linestyles='-')
#     plt.clabel(contour, inline=True, fontsize=10, fmt=f'V = {d_star:.4f}')
#     # d1 = 1.5*d_star
#     # contour = plt.contour(Delta, Omega, V_grid, levels=[d1], colors='red', linewidths=2, linestyles='-')
#     # plt.clabel(contour, inline=True, fontsize=10, fmt=f'V = {d1:.4f}')
#     # 采样椭圆边界（黑色虚线）
#     # theta = np.linspace(0, 2*np.pi, 200)
#     # delta_ellipse = a * np.cos(theta)
#     # omega_ellipse = b * np.sin(theta)
#     # plt.plot(delta_ellipse, omega_ellipse, 'k--', linewidth=1.5, label='Sampling ellipse')
#
#     # 平衡点
#     plt.plot(0, 0, 'ko', markersize=6, label='Equilibrium')
#     plt.plot(delta_uep, 0, 'ko', markersize=6, label='Equilibrium')
#
#     # initial_point = [-2.5, -0.1]
#     # traj = integrate_trajectory(initial_point)
#     # plt.plot(traj[:, 0], traj[:, 1], 'black', linewidth=1.5, alpha=0.8, label='Trajectory')
#     # plt.plot(initial_point[0], initial_point[1], 'black', markersize=8, label='Start point')
#
#     traj1 = backward()
#     plt.plot(traj1[:, 0], traj1[:, 1], 'k-', lw=1.5)
#
#     plt.xlim(x_range)
#     plt.ylim(y_range)
#     plt.xlabel('δ (rad)')
#     plt.ylabel('ω p.u.')
#     plt.title(f'Stability Region (V < {d_star:.6f}) and Sampling Ellipse')
#     plt.grid(alpha=0.3)
#     # plt.legend()
#     plt.tight_layout()
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))
#     plt.show()
# ------------------------------
# 7. 主程序
# ------------------------------
if __name__ == "__main__":
    # 初始化模型和优化器
    model = LyapunovNN(hidden_dim=16)
    # model = ICNN(hidden_dims=[16, 16])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练
    print("开始训练神经 Lyapunov 函数...")
    loss_hist = train_lyapunov(model, optimizer, n_epochs=1500,
                               alpha=0.9, beta=1.0, gamma=0.1, c=0.01)
    # loss_hist = train_lyapunov(model, optimizer, n_epochs=1500,
    #                            alpha=1.1, beta=1.0, gamma=0.01, c=0.01)
    # plt.plot(loss_hist)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()

    # 可视化
    plot_lyapunov_3d(model)
    # d, _, _ = estimate_stability_region(model, u=3.0)
    # plot_stability_region(model, d)
    # d_star, _ = estimate_d_star_on_ellipse(model, a=delta_range, b=omega_range)
    # d_star=0.006833
    delta_val = delta_uep
    omega_val = 0.0
    d_star = compute_V_at_point(model, delta_val, omega_val)
    plot_ROA(model,d_star , a=delta_range, b=omega_range)