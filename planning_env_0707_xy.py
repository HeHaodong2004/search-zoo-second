# domains_cc/path_planning_rl/env/planning_env.py
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot

class PathPlanningMaskEnv(gym.Env):
    """单智能体路径规划，obs里包含action_mask的环境"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scen_file: str,
        problem_index: int,
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 500,
        angle_tol: float = 0.10,
    ):
        super().__init__()
        # --- 地图、碰撞检查器、起终点、动力学、足迹（和原来一样） ---
        map_path = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_path)
        self.world_cc = WorldCollisionChecker(grid, resolution)

        sg_pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = sg_pairs[problem_index]

        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- action / obs 空间 ---
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)

        # state_vec_dim = start_state + goal_state
        dim = self.start_state.size * 2
        # 现在 observation 是个 dict：{"state": Box(dim,), "action_mask": MultiBinary(n_actions)}
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
        })

        # 其他参数
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert self.state is not None, "must reset first"
        cur = self.state.copy()
        next_state = self.dynamics.get_next_states(cur)[action, -1]
        self.state = next_state

        # reward = 距离减少
        prev_d = np.linalg.norm(cur[:2] - self.goal_state[:2])
        new_d  = np.linalg.norm(next_state[:2] - self.goal_state[:2])
        reward = prev_d - new_d

        self._steps += 1
        reached_pos = new_d < 1.0
        angle_err = abs(np.arctan2(
            np.sin(next_state[2]-self.goal_state[2]),
            np.cos(next_state[2]-self.goal_state[2])
        ))
        reached_ang = angle_err < self.angle_tol

        done = reached_pos and reached_ang or (self._steps >= self.max_steps)
        return self._get_obs(), reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 所需：返回哪些动作合法"""
        assert self.state is not None
        cur = self.state.copy()
        # paths: (n_actions, steps+1, vS), 取前3维做碰撞
        traj = self.dynamics.get_next_states(cur)[:, :, :3].copy(order="C")
        mask = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            mask[i] = self.world_cc.isValid(self.footprint, traj[i]).all()
        return mask


    def _get_obs(self) -> Dict:
        """把 state + mask 打包成 dict"""
        st = np.concatenate([self.state, self.goal_state]).astype(np.float32)
        return {
            "state": st,
            "action_mask": self.action_masks(),
        }

    def render(self, mode="human"):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax; ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        cur_xytheta = self.state[:3]
        addXYThetaToPlot(self.world_cc, ax, self.footprint, cur_xytheta)
        ax.set_title(f"Step {self._steps}")

        if mode=="rgb_array":
            self._fig.canvas.draw()
            w,h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            plt.pause(1/self.metadata["render_fps"])

    def close(self):
        if self._fig:
            plt.close(self._fig)
            self._fig = self._ax = None
