# domains_cc/path_planning_rl/env/planning_env.py
# -*- coding: utf-8 -*-
import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        max_scan_dist: float = 10.0,  # in meters
        hist_len: int = 6,
    ):
        super().__init__()

        # --- load dynamics & footprint ---
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        # --- load and inflate map ---
        map_path = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_path)
        fp_cfg = yaml.safe_load(open(footprint_config, encoding='utf-8'))
        radius = fp_cfg.get("radius", 0.5)
        r_cells = int(np.ceil(radius / resolution))
        struct = np.zeros((2 * r_cells + 1, 2 * r_cells + 1), dtype=bool)
        for i in range(2 * r_cells + 1):
            for j in range(2 * r_cells + 1):
                if (i - r_cells) ** 2 + (j - r_cells) ** 2 <= r_cells ** 2:
                    struct[i, j] = True
        obstacles = (grid != 0)
        self.inflated = binary_dilation(obstacles, structure=struct)
        self.resolution = resolution
        self.max_scan_dist = max_scan_dist
        self.world_cc = WorldCollisionChecker(grid, resolution)

        # --- start / goal ---
        sg_pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = sg_pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- global distance map & gradient for shaping ---
        self.dist_map = self._compute_distance_map(grid, self.goal_xytheta[:2], resolution)
        finite = np.isfinite(self.dist_map)
        if np.any(finite):
            maxd = float(np.nanmax(self.dist_map[finite]))
        else:
            maxd = 0.0
        self.dist_map[~finite] = maxd

        gy, gx = np.gradient(self.dist_map, edge_order=2)
        mag = np.hypot(gx, gy)
        self.grad_x = np.zeros_like(gx, dtype=np.float32)
        self.grad_y = np.zeros_like(gy, dtype=np.float32)
        nz = mag > 1e-8
        self.grad_x[nz] = -gx[nz] / mag[nz]
        self.grad_y[nz] = -gy[nz] / mag[nz]

        # shaping parameters
        self.gamma_shaping = 0.99
        self.time_cost = -0.1
        self.completion_bonus = 10.0

        # --- action space & history ---
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        self.hist_len = hist_len
        self.action_history = deque(maxlen=self.hist_len)

        # --- laser setup ---
        self.n_lasers = n_lasers
        self.laser_angles = np.linspace(-np.pi, np.pi, n_lasers, endpoint=False)

        # --- observation space ---
        self.observation_space = spaces.Dict({
            "scan":        spaces.Box(0.0, max_scan_dist, (n_lasers,), dtype=np.float32),
            "goal_vec":    spaces.Box(-np.inf, np.inf, (2,),          dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
            "hist":        spaces.MultiDiscrete([self.n_actions] * self.hist_len),
        })

        # --- internal counters ---
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        return self._get_obs(), {}

    '''def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        # compute potential φ(s)
        phi_cur = self._interpolate_dist(cur[:2])

        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()
        if not valid:
            return self._get_obs(), -10.0, True, False, {"collision": True}

        # apply action
        self.state = traj[-1]
        self.action_history.append(action)

        # base reward: distance improvement
        prev_d = np.linalg.norm(cur[:2] - self.goal_state[:2])
        new_d  = np.linalg.norm(self.state[:2] - self.goal_state[:2])
        reward = prev_d - new_d

        # time cost
        reward += self.time_cost

        # shaping: γ φ(s') − φ(s)
        phi_new = self._interpolate_dist(self.state[:2])
        reward += (self.gamma_shaping * phi_new - phi_cur)

        # completion bonus
        self._steps += 1
        reached = (new_d < 1.0 and
                   abs(np.arctan2(
                       np.sin(self.state[2] - self.goal_state[2]),
                       np.cos(self.state[2] - self.goal_state[2])
                   )) < self.angle_tol)
        if reached:
            reward += self.completion_bonus

        done = reached or (self._steps >= self.max_steps)
        return self._get_obs(), reward, done, False, {}'''
    def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        # compute potential φ(s)
        phi_cur = self._interpolate_dist(cur[:2])

        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()
        if not valid:
            return self._get_obs(), -10.0, True, False, {"collision": True, "reached": False}

        # apply action
        self.state = traj[-1]
        self.action_history.append(action)

        # base reward: distance improvement
        prev_d = np.linalg.norm(cur[:2] - self.goal_state[:2])
        new_d  = np.linalg.norm(self.state[:2] - self.goal_state[:2])
        reward = prev_d - new_d

        # time cost
        reward += self.time_cost

        # shaping: γ φ(s') − φ(s)
        phi_new = self._interpolate_dist(self.state[:2])
        #reward += (self.gamma_shaping * phi_new - phi_cur)
        reward += (phi_cur - self.gamma_shaping * phi_new)

        # completion bonus
        self._steps += 1
        reached = (
            new_d < 1.0
            and abs(np.arctan2(
                np.sin(self.state[2] - self.goal_state[2]),
                np.cos(self.state[2] - self.goal_state[2])
            )) < self.angle_tol
        )
        if reached:
            reward += self.completion_bonus

        done = reached or (self._steps >= self.max_steps)
        info = {"collision": False, "reached": reached}
        return self._get_obs(), reward, done, False, info


    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        masks = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            masks[i] = self.world_cc.isValid(self.footprint, trajs[i]).all()
        return masks

    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]

        # laser scans
        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        max_steps = int(self.max_scan_dist / self.resolution)
        H, W = self.inflated.shape
        for idx, ang in enumerate(self.laser_angles):
            dx = np.cos(theta + ang)
            dy = np.sin(theta + ang)
            for s in range(max_steps):
                px = x + dx * s * self.resolution
                py = y + dy * s * self.resolution
                i = int(py / self.resolution)
                j = int(px / self.resolution)
                if i < 0 or i >= H or j < 0 or j >= W or self.inflated[i, j]:
                    scans[idx] = s * self.resolution
                    break

        # goal vector in robot frame
        dxg = self.goal_state[0] - x
        dyg = self.goal_state[1] - y
        gx = dxg * np.cos(-theta) - dyg * np.sin(-theta)
        gy = dxg * np.sin(-theta) + dyg * np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        hist = np.array(self.action_history, dtype=np.int64)

        return {
            "scan":        scans,
            "goal_vec":    goal_vec,
            "action_mask": self.action_masks(),
            "hist":        hist,
        }

    def render(self, mode="human"):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax; ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])
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
        """
        BFS from the cell nearest goal_xy, computing cell‐distance to every free cell.
        """
        from collections import deque
        H, W = grid.shape
        free = (grid == 0)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        # find nearest free cell to goal
        gj0 = int(round(goal_xy[0] / resolution))
        gi0 = int(round(goal_xy[1] / resolution))
        gi0, gj0 = np.clip(gi0, 0, H - 1), np.clip(gj0, 0, W - 1)
        if not free[gi0, gj0]:
            visited = np.zeros_like(free, bool)
            dq = deque([(gi0, gj0)]); visited[gi0, gj0] = True
            neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while dq:
                i, j = dq.popleft()
                for di, dj in neighs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                        if free[ni, nj]:
                            gi0, gj0 = ni, nj
                            dq.clear()
                            break
                        visited[ni, nj] = True
                        dq.append((ni, nj))
        dist[gi0, gj0] = 0.0

        # BFS flood
        dq = deque([(gi0, gj0)])
        neighs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while dq:
            i, j = dq.popleft()
            for di, dj in neighs:
                ni, nj = i + di, j + dj
                if (0 <= ni < H and 0 <= nj < W and
                    free[ni, nj] and not np.isfinite(dist[ni, nj])):
                    dist[ni, nj] = dist[i, j] + 1.0
                    dq.append((ni, nj))
        return dist

    def _interpolate_dist(self, xy: np.ndarray) -> float:
        """
        Bilinear interpolate self.dist_map at continuous (x,y).
        """
        x, y = xy
        H, W = self.dist_map.shape
        i_f = np.clip(y / self.resolution, 0.0, H - 1 - 1e-6)
        j_f = np.clip(x / self.resolution, 0.0, W - 1 - 1e-6)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = i0 + 1, j0 + 1
        di, dj = i_f - i0, j_f - j0

        d00 = self.dist_map[i0, j0]
        d10 = self.dist_map[i1, j0]
        d01 = self.dist_map[i0, j1]
        d11 = self.dist_map[i1, j1]

        return (
            d00 * (1 - di) * (1 - dj)
            + d10 * di * (1 - dj)
            + d01 * (1 - di) * dj
            + d11 * di * dj
        )
