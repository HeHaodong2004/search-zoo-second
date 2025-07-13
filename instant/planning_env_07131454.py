import yaml
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_dilation
from collections import deque
from typing import Optional, Dict, Tuple
import numpy as np

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot


class PathPlanningMaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scen_file: str,
        problem_index: int,
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 500,
        angle_tol: float = 0.10,
        n_lasers: int = 16,
        max_scan_dist: float = 10.0,
        hist_len: int = 6,
    ):
        super().__init__()

        # --- dynamics & footprint ---
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        # --- load & inflate map ---
        map_path = get_map_file_path(scen_file)
        raw_grid, resolution = parse_map_file(map_path)
        self.resolution = resolution
        '''fp_cfg = yaml.safe_load(open(footprint_config, encoding='utf-8'))
        r = fp_cfg.get("radius", 0.5)
        r_cells = int(np.ceil(r / resolution))
        struct = np.zeros((2*r_cells+1, 2*r_cells+1), bool)
        for i in range(2*r_cells+1):
            for j in range(2*r_cells+1):
                if (i-r_cells)**2 + (j-r_cells)**2 <= r_cells**2:
                    struct[i, j] = True
        # inflated obstacles mask: True where blocked
        self.inflated = binary_dilation((raw_grid != 0), structure=struct)'''
        self.inflated = (raw_grid != 0)
        self.max_scan_dist = max_scan_dist
        self.world_cc = WorldCollisionChecker(raw_grid, resolution)

        # --- start / goal ---
        pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- compute distance map on inflated map ---
        grid_bin = (~self.inflated).astype(np.uint8)   # 1 表示 free，0 表示 obstacle
        # 1) BFS 只在 grid_bin == 1 区域跑，unreachable 保留 inf
        self.dist_map = self._compute_distance_map(grid_bin,
                                                   self.goal_xytheta[:2],
                                                   self.resolution)
        # 2) 标记 unreachable（inf），再计算 max_potential
        unreachable = np.isinf(self.dist_map)
        reachable   = ~unreachable
        if np.any(reachable):
            # 只在可达区域里取 max
            self.max_potential = float(np.max(self.dist_map[reachable]))
        else:
            self.max_potential = 0.0
        # 3) 把 unreachable 填成 max_potential
        self.dist_map[unreachable] = self.max_potential

        # --- compute unit gradient field ---
        # 把 resolution 作为 spacing，这样梯度单位是 1/m
        gy, gx = np.gradient(self.dist_map, self.resolution, edge_order=2)
        mag = np.hypot(gx, gy)
        self.grad_x = np.zeros_like(gx, dtype=np.float32)
        self.grad_y = np.zeros_like(gy, dtype=np.float32)
        nz = mag > 1e-8
        # 负号保留，指向距离减小最快的方向（也就是朝目标）
        self.grad_x[nz] = -gx[nz] / mag[nz]
        self.grad_y[nz] = -gy[nz] / mag[nz]

        # --- 用同一个 mask，把 obstacle 或 unreachable 的梯度都设为 0 ---
        mask = (grid_bin == 1) & (~unreachable)   # True 表示“真正 free 且可达”
        self.grad_x[~mask] = 0.0
        self.grad_y[~mask] = 0.0


        # --- reward weights ---
        self.align_coeff      =  1.0    # weight for gradient alignment reward
        self.time_cost        = -0.1    # per‐step penalty
        self.completion_bonus = 50.0
        self.collision_penalty= -10.0

        # --- action space & history ---
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        self.hist_len = hist_len
        self.action_history = deque(maxlen=hist_len)

        # --- lidar setup ---
        self.n_lasers = n_lasers
        self.laser_angles = np.linspace(-np.pi, np.pi, n_lasers, endpoint=False)

        # --- observation space ---
        self.observation_space = spaces.Dict({
            "scan":        spaces.Box(0.0, max_scan_dist, (n_lasers,), dtype=np.float32),
            "goal_vec":    spaces.Box(-np.inf, np.inf, (2,),        dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
            "hist":        spaces.MultiDiscrete([self.n_actions]*hist_len),
            "dist_grad":   spaces.Box(-1.0, 1.0, (2,),            dtype=np.float32),
            "dist_phi":    spaces.Box(0.0, 1.0, (1,),              dtype=np.float32),
        })

        # --- internal state ---
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax = None

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        # simulate action
        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        if not self.world_cc.isValid(self.footprint, traj).all():
            # collision: heavy penalty + done
            return self._get_obs(), self.collision_penalty, True, False, {
                "collision": True, "reached": False
            }

        # accept new state
        self.state = traj[-1]
        self.action_history.append(action)

        # 1) compute world‐frame unit gradient at the _old_ position
        gx = self._interpolate_grad(self.grad_x, cur[:2])
        gy = self._interpolate_grad(self.grad_y, cur[:2])
        grad_vec = np.array([gx, gy], dtype=np.float32)
        # (already unit‐length because we normalized in __init__)

        # 2) compute motion direction unit vector
        dx = self.state[0] - cur[0]
        dy = self.state[1] - cur[1]
        dist = np.hypot(dx, dy)
        if dist > 1e-8:
            motion_dir = np.array([dx, dy], dtype=np.float32) / dist
        else:
            motion_dir = np.zeros(2, dtype=np.float32)

        # 3) alignment reward = cosine of angle between motion & gradient
        align = float(np.dot(motion_dir, grad_vec))
        reward = self.align_coeff * align

        # 4) small time penalty to encourage speed
        reward += self.time_cost

        # 5) completion bonus
        self._steps += 1
        dxg = self.state[0] - self.goal_state[0]
        dyg = self.state[1] - self.goal_state[1]
        ang_diff = abs(np.arctan2(
            np.sin(self.state[2] - self.goal_state[2]),
            np.cos(self.state[2] - self.goal_state[2])
        ))
        reached = (np.hypot(dxg, dyg) < 1.0 and ang_diff < self.angle_tol)
        if reached:
            reward += self.completion_bonus

        # 6) done if goal or max steps
        done = reached or (self._steps >= self.max_steps)
        info = {"collision": False, "reached": reached}
        return self._get_obs(), reward, done, False, info


    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        mask = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            mask[i] = self.world_cc.isValid(self.footprint, trajs[i]).all()
        return mask

    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]

        # lidar scans
        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        H, W = self.inflated.shape
        max_steps = int(self.max_scan_dist / self.resolution)
        for idx, ang in enumerate(self.laser_angles):
            dx, dy = np.cos(theta+ang), np.sin(theta+ang)
            for s in range(max_steps):
                px, py = x + dx*s*self.resolution, y + dy*s*self.resolution
                i, j = int(py/self.resolution), int(px/self.resolution)
                if not (0<=i<H and 0<=j<W) or self.inflated[i,j]:
                    scans[idx] = s*self.resolution
                    break

        # goal vector in robot frame
        dxg, dyg = self.goal_state[0]-x, self.goal_state[1]-y
        gx =  dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy =  dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        # gradient interpolation (world)
        gx_w = self._interpolate_grad(self.grad_x, self.state[:2])
        gy_w = self._interpolate_grad(self.grad_y, self.state[:2])
        # rotate gradient into robot frame
        gx_r =  gx_w * np.cos(-theta) - gy_w * np.sin(-theta)
        gy_r =  gx_w * np.sin(-theta) + gy_w * np.cos(-theta)

        # potential interpolation and normalize
        phi = self._interpolate_dist(self.state[:2])
        phi_norm = phi / (self.max_potential + 1e-6)

        hist = np.array(self.action_history, dtype=np.int64)

        return {
            "scan":        scans,
            "goal_vec":    goal_vec,
            "action_mask": self.action_masks(),
            "hist":        hist,
            "dist_grad":   np.array([gx_r, gy_r], dtype=np.float32),
            "dist_phi":    np.array([phi_norm], dtype=np.float32),
        }

    def render(self, mode="human"):
            if self._fig is None:
                import matplotlib.pyplot as plt
                self._fig, self._ax = plt.subplots(figsize=(6,6))
            ax = self._ax; ax.clear()

            # --- 画栅格、起点/终点/当前状态 ---
            addGridToPlot(self.world_cc, ax)
            addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
            addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
            addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])

            # --- 画潜在场距离图（可选） ---
            # ax.imshow(self.dist_map, origin='lower',
            #           extent=[0, self.dist_map.shape[1]*self.resolution,
            #                   0, self.dist_map.shape[0]*self.resolution],
            #           cmap='gray', alpha=0.3)

            # --- 在这里加上潜在场梯度箭头（采样显示，避免过密） ---
            import numpy as np
            H, W = self.grad_x.shape
            step = max(1, min(H, W)//20)
            i_idx = np.arange(0, H, step)
            j_idx = np.arange(0, W, step)
            jj, ii = np.meshgrid(j_idx, i_idx)
            X = (jj + 0.5) * self.resolution
            Y = (ii + 0.5) * self.resolution
            U = self.grad_x[ii, jj]
            V = self.grad_y[ii, jj]

            # **固定坐标系、等比例、反转 y 轴**
            ax.set_xlim(0, W * self.resolution)
            ax.set_ylim(0, H * self.resolution)
            #ax.invert_yaxis()
            ax.set_aspect('equal')

            # **用真实向量长度，不做额外缩放**
            ax.quiver(
                X, Y, U, V,
                angles='xy', scale_units='xy', scale=5,
                width=0.003, alpha=0.8
            )
            # --- 画距离场热力图 ---
            # 将 dist_map 归一化到 [0,1]，并用 seaborn/cmap 渐变
            import matplotlib.pyplot as plt
            dm = self.dist_map.copy()
            # 把最大值归一化
            dm = dm / (np.max(dm) + 1e-6)
            ax.imshow(
                np.flipud(dm),
                origin='lower',
                extent=[0, dm.shape[1]*self.resolution,
                        0, dm.shape[0]*self.resolution],
                cmap='viridis',   # 也可选 'hot' 或其他
                alpha=0.5         # 透明度，根据需要调整
            )

            ax.set_title(f"Step {self._steps}")
            if mode == "rgb_array":
                self._fig.canvas.draw()
                w, h = self._fig.canvas.get_width_height()
                buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
                return buf.reshape(h, w, 3)
            else:
                plt.pause(1 / self.metadata["render_fps"])



    @staticmethod
    def _compute_distance_map(grid: np.ndarray,
                              goal_xy: np.ndarray,
                              resolution: float) -> np.ndarray:
        from collections import deque
        H, W = grid.shape
        free = (grid != 0)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        # find nearest free cell to goal
        gj0 = int(round(goal_xy[0] / resolution))
        gi0 = int(round(goal_xy[1] / resolution))
        gi0, gj0 = np.clip(gi0, 0, H-1), np.clip(gj0, 0, W-1)
        if not free[gi0, gj0]:
            visited = np.zeros_like(free, bool)
            dq = deque([(gi0, gj0)]); visited[gi0, gj0] = True
            neighs = [(-1,0),(1,0),(0,-1),(0,1)]
            while dq:
                i,j = dq.popleft()
                for di,dj in neighs:
                    ni,nj = i+di, j+dj
                    if 0<=ni<H and 0<=nj<W and not visited[ni,nj]:
                        if free[ni,nj]:
                            gi0,gj0 = ni,nj; dq.clear(); break
                        visited[ni,nj]=True; dq.append((ni,nj))
        dist[gi0, gj0] = 0.0

        dq = deque([(gi0, gj0)])
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            i,j = dq.popleft()
            for di,dj in neighs:
                ni,nj = i+di, j+dj
                if (0<=ni<H and 0<=nj<W and free[ni,nj] and not np.isfinite(dist[ni,nj])):
                    dist[ni, nj] = dist[i, j] + 1.0
                    dq.append((ni, nj))
        return dist

    def _interpolate_dist(self, xy: np.ndarray) -> float:
        x, y = xy
        H, W = self.dist_map.shape
        i_f = np.clip(y / self.resolution, 0.0, H-1-1e-6)
        j_f = np.clip(x / self.resolution, 0.0, W-1-1e-6)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1,H-1), min(j0+1,W-1)
        di, dj = i_f - i0, j_f - j0
        d00 = self.dist_map[i0,j0]; d10 = self.dist_map[i1,j0]
        d01 = self.dist_map[i0,j1]; d11 = self.dist_map[i1,j1]
        return (
            d00*(1-di)*(1-dj) + d10*di*(1-dj)
            + d01*(1-di)*dj + d11*di*dj
        )

    def _interpolate_grad(self, grad_map: np.ndarray, xy: np.ndarray) -> float:
        x, y = xy
        H, W = grad_map.shape
        i_f = np.clip(y / self.resolution, 0.0, H-1-1e-6)
        j_f = np.clip(x / self.resolution, 0.0, W-1-1e-6)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1,H-1), min(j0+1,W-1)
        di, dj = i_f - i0, j_f - j0
        g00 = grad_map[i0,j0]; g10 = grad_map[i1,j0]
        g01 = grad_map[i0,j1]; g11 = grad_map[i1,j1]
        return (
            g00*(1-di)*(1-dj) + g10*di*(1-dj)
            + g01*(1-di)*dj + g11*di*dj
        )
