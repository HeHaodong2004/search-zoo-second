import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import binary_dilation
from collections import deque
from typing import Optional, Dict, Tuple

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
        fp_cfg = yaml.safe_load(open(footprint_config, encoding='utf-8'))
        r = fp_cfg.get("radius", 0.5)
        r_cells = int(np.ceil(r / resolution))
        struct = np.zeros((2*r_cells+1, 2*r_cells+1), bool)
        for i in range(2*r_cells+1):
            for j in range(2*r_cells+1):
                if (i-r_cells)**2 + (j-r_cells)**2 <= r_cells**2:
                    struct[i, j] = True
        # inflated obstacles mask: True where blocked
        self.inflated = binary_dilation((raw_grid != 0), structure=struct)
        self.max_scan_dist = max_scan_dist
        self.world_cc = WorldCollisionChecker(raw_grid, resolution)

        # --- start / goal ---
        pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- compute distance map on inflated map ---
        grid2 = self.inflated.astype(np.uint8)
        self.dist_map = self._compute_distance_map(grid2, self.goal_xytheta[:2], resolution)
        finite = np.isfinite(self.dist_map)
        if np.any(finite):
            self.max_potential = float(np.nanmax(self.dist_map[finite]))
        else:
            self.max_potential = 0.0
        # fill unreachable with max
        self.dist_map[~finite] = self.max_potential

        # compute unit gradient field
        gy, gx = np.gradient(self.dist_map, edge_order=2)
        mag = np.hypot(gx, gy)
        self.grad_x = np.zeros_like(gx, dtype=np.float32)
        self.grad_y = np.zeros_like(gy, dtype=np.float32)
        nz = mag > 1e-8
        self.grad_x[nz] = -gx[nz] / mag[nz]
        self.grad_y[nz] = -gy[nz] / mag[nz]

        # --- reward weights ---
        self.gamma_shaping     = 0.99
        self.time_cost        = -0.5
        self.completion_bonus  = 50.0
        self.collision_penalty = -10.0

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

        # potential-based shaping
        phi_cur = self._interpolate_dist(cur[:2])

        # simulate action
        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        if not self.world_cc.isValid(self.footprint, traj).all():
            return self._get_obs(), self.collision_penalty, True, False, {"collision": True, "reached": False}

        # update state
        self.state = traj[-1]
        self.action_history.append(action)

        # shaping reward: gamma*phi' - phi
        phi_new = self._interpolate_dist(self.state[:2])
        reward = self.gamma_shaping * phi_new - phi_cur

        # time penalty
        reward += self.time_cost

        # completion bonus
        self._steps += 1
        dx = self.state[0] - self.goal_state[0]
        dy = self.state[1] - self.goal_state[1]
        ang_diff = abs(np.arctan2(np.sin(self.state[2]-self.goal_state[2]),
                                  np.cos(self.state[2]-self.goal_state[2])))
        reached = (np.hypot(dx, dy) < 1.0 and ang_diff < self.angle_tol)
        if reached:
            reward += self.completion_bonus

        done = reached or (self._steps >= self.max_steps)
        return self._get_obs(), reward, done, False, {"collision": False, "reached": reached}

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
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])
        ax.set_title(f"Step {self._steps}")
        if mode=="rgb_array":
            self._fig.canvas.draw()
            w,h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            import matplotlib.pyplot as plt
            plt.pause(1/self.metadata["render_fps"])

    @staticmethod
    def _compute_distance_map(grid: np.ndarray,
                              goal_xy: np.ndarray,
                              resolution: float) -> np.ndarray:
        from collections import deque
        H, W = grid.shape
        free = (grid == 0)
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
