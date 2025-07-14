# planning_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
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

        # --- load raw map (top-origin) ---
        map_path = get_map_file_path(scen_file)
        raw_grid, resolution = parse_map_file(map_path)  # bool array, True=obs
        self.resolution = resolution
        self.max_scan_dist = max_scan_dist

        # --- collision checker wants raw_grid; it flips+transposes internally ---
        self.world_cc = WorldCollisionChecker(raw_grid, resolution)

        # --- bottom-origin grid for planning & rendering ---
        bottom_grid = np.flipud(raw_grid)               # row 0 = world y=0
        free_mask   = (~bottom_grid).astype(np.uint8)   # 1=free, 0=obs
        H, W        = bottom_grid.shape

        # --- start / goal (world coords, origin bottom-left) ---
        pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- goal in grid indices ---
        gx = int(np.floor(self.goal_xytheta[0] / resolution))
        gy = int(np.floor(self.goal_xytheta[1] / resolution))

        # --- distance map on bottom-origin grid ---
        dist_arr = self._compute_distance_map(free_mask, np.array([gx, gy]), resolution)
        unreachable = np.isinf(dist_arr)
        reachable   = ~unreachable
        self.max_potential = float(np.max(dist_arr[reachable])) if np.any(reachable) else 0.0
        dist_arr[unreachable] = self.max_potential
        self.dist_map = dist_arr  # bottom-origin

        # --- gradient field (unit vectors) ---
        gy_e, gx_e = np.gradient(self.dist_map, resolution, edge_order=2)
        mag = np.hypot(gx_e, gy_e)
        self.grad_x = np.zeros_like(gx_e, dtype=np.float32)
        self.grad_y = np.zeros_like(gy_e, dtype=np.float32)
        nz = mag > 1e-8
        self.grad_x[nz] = -gx_e[nz] / mag[nz]
        self.grad_y[nz] = -gy_e[nz] / mag[nz]
        valid = (free_mask == 1) & (~unreachable)
        self.grad_x[~valid] = 0.0
        self.grad_y[~valid] = 0.0

        # --- reward weights ---
        self.align_coeff       =  1.0
        self.time_cost         = -0.1
        self.completion_bonus  = 50.0
        self.collision_penalty = -10.0

        # --- action & observation spaces ---
        self.n_actions      = self.dynamics.motion_primitives.shape[0]
        self.action_space   = spaces.Discrete(self.n_actions)
        self.hist_len       = hist_len
        self.action_history = deque(maxlen=hist_len)

        # --- lidar ---
        self.n_lasers     = n_lasers
        self.laser_angles = np.linspace(-np.pi, np.pi, n_lasers, endpoint=False)

        self.observation_space = spaces.Dict({
            "scan":       spaces.Box(0.0, max_scan_dist, (n_lasers,), dtype=np.float32),
            "goal_vec":   spaces.Box(-np.inf, np.inf, (2,),       dtype=np.float32),
            "action_mask":spaces.MultiBinary(self.n_actions),
            "hist":       spaces.MultiDiscrete([self.n_actions]*hist_len),
            "dist_grad":  spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "dist_phi":   spaces.Box(0.0, 1.0,   (1,), dtype=np.float32),
        })

        # --- internal state ---
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax  = None


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
        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        if not self.world_cc.isValid(self.footprint, traj).all():
            return self._get_obs(), self.collision_penalty, True, False, {
                "collision": True, "reached": False
            }

        self.state = traj[-1]
        self.action_history.append(action)

        # gradient alignment reward
        gx = self._interpolate_grad(self.grad_x, cur[:2])
        gy = self._interpolate_grad(self.grad_y, cur[:2])
        grad_vec = np.array([gx, gy], dtype=np.float32)

        dx = self.state[0] - cur[0]
        dy = self.state[1] - cur[1]
        dist = np.hypot(dx, dy)
        motion_dir = np.array([dx, dy], dtype=np.float32) / (dist + 1e-8)
        reward = self.align_coeff * float(np.dot(motion_dir, grad_vec)) + self.time_cost

        # completion bonus
        self._steps += 1
        dxg = self.state[0] - self.goal_state[0]
        dyg = self.state[1] - self.goal_state[1]
        reached = (np.hypot(dxg, dyg) < 1.0)
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

        # lidar
        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        H, W = self.dist_map.shape
        max_steps = int(self.max_scan_dist / self.resolution)
        for idx, ang in enumerate(self.laser_angles):
            dx = np.cos(theta+ang); dy = np.sin(theta+ang)
            for s in range(max_steps):
                px = x + dx*s*self.resolution
                py = y + dy*s*self.resolution
                ix = int(np.floor(px/self.resolution))
                iy = int(np.floor(py/self.resolution))
                if not (0<=ix<W and 0<=iy<H) or self.world_cc.grid[ix,iy] != 0:
                    scans[idx] = s*self.resolution
                    break

        # goal vector in robot frame
        dxg = self.goal_state[0]-x; dyg = self.goal_state[1]-y
        gx =  dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy =  dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        # dist-gradient in robot frame
        gx_w = self._interpolate_grad(self.grad_x, self.state[:2])
        gy_w = self._interpolate_grad(self.grad_y, self.state[:2])
        gx_r = gx_w*np.cos(-theta) - gy_w*np.sin(-theta)
        gy_r = gx_w*np.sin(-theta) + gy_w*np.cos(-theta)

        # normalized distance
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
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax
        ax.clear()

        # obstacles, start, goal, robot
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])

        # heatmap (bottom-origin!)
        H, W = self.dist_map.shape
        dm = self.dist_map / (self.max_potential + 1e-6)
        ax.imshow(
            dm,
            origin='lower',
            extent=[0, W*self.resolution, 0, H*self.resolution],
            cmap='viridis',
            alpha=0.5
        )

        # arrows (same bottom-origin frame)
        step = max(1, min(H, W)//50)
        rows = np.arange(0, H, step)
        cols = np.arange(0, W, step)
        xs = (cols+0.5)*self.resolution
        ys = (rows+0.5)*self.resolution
        Xs, Ys = np.meshgrid(xs, ys, indexing='xy')
        U = self.grad_x[np.ix_(rows, cols)]
        V = self.grad_y[np.ix_(rows, cols)]

        ax.quiver(
            Xs, Ys, U, V,
            angles='xy', scale_units='xy', scale=2,
            width=0.002, alpha=0.8
        )

        ax.set_xlim(0, W*self.resolution)
        ax.set_ylim(0, H*self.resolution)
        ax.set_aspect('equal')
        ax.set_title(f"Step {self._steps}")

        if mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            plt.pause(1/self.metadata["render_fps"])


    @staticmethod
    def _compute_distance_map(grid: np.ndarray, goal_xy: np.ndarray, resolution: float) -> np.ndarray:
        from collections import deque
        H, W = grid.shape
        free = (grid == 1)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        gx, gy = goal_xy.astype(int)
        # if goal on obstacle, find nearest free
        if not free[gx, gy]:
            visited = np.zeros_like(free, bool)
            dq = deque([(gx, gy)])
            visited[gx, gy] = True
            neighs = [(-1,0),(1,0),(0,-1),(0,1)]
            while dq:
                x0, y0 = dq.popleft()
                for dx, dy in neighs:
                    xn, yn = x0+dx, y0+dy
                    if 0<=xn<H and 0<=yn<W and not visited[xn,yn]:
                        if free[xn,yn]:
                            gx, gy = xn, yn
                            dq.clear()
                            break
                        visited[xn,yn] = True
                        dq.append((xn,yn))

        dist[gx, gy] = 0.0
        dq = deque([(gx, gy)])
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            x0, y0 = dq.popleft()
            for dx, dy in neighs:
                xn, yn = x0+dx, y0+dy
                if (0<=xn<H and 0<=yn<W and free[xn,yn]
                        and not np.isfinite(dist[xn,yn])):
                    dist[xn,yn] = dist[x0,y0] + 1.0
                    dq.append((xn,yn))
        return dist


    def _interpolate_dist(self, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(x/self.resolution, 0, self.dist_map.shape[0]-1)
        j_f = np.clip(y/self.resolution, 0, self.dist_map.shape[1]-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.dist_map.shape[0]-1), min(j0+1, self.dist_map.shape[1]-1)
        di, dj = i_f - i0, j_f - j0
        d00 = self.dist_map[i0, j0]; d10 = self.dist_map[i1, j0]
        d01 = self.dist_map[i0, j1]; d11 = self.dist_map[i1, j1]
        return (
            d00*(1-di)*(1-dj)
          + d10*di*(1-dj)
          + d01*(1-di)*dj
          + d11*di*dj
        )


    def _interpolate_grad(self, grad_map: np.ndarray, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(x/self.resolution, 0, grad_map.shape[0]-1)
        j_f = np.clip(y/self.resolution, 0, grad_map.shape[1]-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, grad_map.shape[0]-1), min(j0+1, grad_map.shape[1]-1)
        di, dj = i_f - i0, j_f - j0
        g00 = grad_map[i0, j0]; g10 = grad_map[i1, j0]
        g01 = grad_map[i0, j1]; g11 = grad_map[i1, j1]
        return (
            g00*(1-di)*(1-dj)
          + g10*di*(1-dj)
          + g01*(1-di)*dj
          + g11*di*dj
        )
