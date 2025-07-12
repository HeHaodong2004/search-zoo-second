import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
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
    ):
        super().__init__()
        # --- load raw map and inflate by robot radius ---
        map_path = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_path)
        fp_cfg = yaml.safe_load(open(footprint_config, encoding='utf-8'))
        radius = fp_cfg.get('radius', 0.5)
        r_cells = int(np.ceil(radius / resolution))
        struct = np.zeros((2*r_cells+1, 2*r_cells+1), dtype=bool)
        for i in range(2*r_cells+1):
            for j in range(2*r_cells+1):
                if (i-r_cells)**2 + (j-r_cells)**2 <= r_cells**2:
                    struct[i,j] = True
        obstacles = (grid != 0)
        self.inflated = binary_dilation(obstacles, structure=struct)
        self.resolution = resolution
        self.max_scan_dist = max_scan_dist
        self.world_cc = WorldCollisionChecker(grid, resolution)

        # --- load dynamics and footprint before using them ---
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        # --- start and goal states ---
        sg_pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = sg_pairs[problem_index]
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # action space
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)

        # laser configuration
        self.n_lasers = n_lasers
        self.laser_angles = np.linspace(-np.pi, np.pi, n_lasers, endpoint=False)

        # observation: scan + goal vector + mask
        self.observation_space = spaces.Dict({
            'scan': spaces.Box(0.0, max_scan_dist, (n_lasers,), dtype=np.float32),
            'goal_vec': spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            'action_mask': spaces.MultiBinary(self.n_actions),
        })

        # internal state
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert self.state is not None, "Call reset() before step()"
        cur = self.state.copy()
        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()
        if not valid:
            return self._get_obs(), -10.0, True, False, {'collision': True}
        self.state = traj[-1]
        prev_d = np.linalg.norm(cur[:2] - self.goal_state[:2])
        new_d  = np.linalg.norm(self.state[:2] - self.goal_state[:2])
        reward = prev_d - new_d
        self._steps += 1
        reached = (new_d < 1.0 and
                   abs(np.arctan2(np.sin(self.state[2]-self.goal_state[2]),
                                 np.cos(self.state[2]-self.goal_state[2]))) < self.angle_tol)
        done = reached or (self._steps >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        masks = np.array([self.world_cc.isValid(self.footprint, t).all() for t in trajs], dtype=bool)
        return masks

    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]
        # laser scan
        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        max_steps = int(self.max_scan_dist / self.resolution)
        H, W = self.inflated.shape
        for idx, ang in enumerate(self.laser_angles):
            dx, dy = np.cos(theta+ang), np.sin(theta+ang)
            for s in range(max_steps):
                px, py = x + dx*s*self.resolution, y + dy*s*self.resolution
                i, j = int(py/self.resolution), int(px/self.resolution)
                if i<0 or i>=H or j<0 or j>=W or self.inflated[i,j]:
                    scans[idx] = s * self.resolution
                    break
        # goal vector in robot frame
        dxg = self.goal_state[0] - x
        dyg = self.goal_state[1] - y
        gx = dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy = dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)
        return {'scan': scans,
                'goal_vec': goal_vec,
                'action_mask': self.action_masks()}

    def render(self, mode="human"):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax; ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])
        ax.set_title(f"Step {self._steps}")
        if mode == "rgb_array":
            self._fig.canvas.draw()
            w,h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            plt.pause(1/self.metadata["render_fps"])
