 # domains_cc/path_planning_rl/env/planning_env.py
# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")                   # 头less 训练也能调用 render
import matplotlib.pyplot as plt
from typing import Optional, Dict 

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot


class PathPlanningMaskEnv(gym.Env):
    """带动作掩码的单机器人路径规划环境。"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    # --------------------------------------------------------------------- #

    def __init__(
        self,
        scen_file: str,
        problem_index: int,
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 500,
        angle_tol: float = 0.10,       # rad ≈ 5.7°
    ):
        # 1) 地图
        map_path = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_path)
        self.world_cc = WorldCollisionChecker(grid, resolution)

        # 2) 起终点 (x, y, θ)
        sg_pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = sg_pairs[problem_index]

        # 3) 动力学 & 足迹
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # 4) Gymnasium spaces
        self.n_actions     = self.dynamics.motion_primitives.shape[0]
        self.action_space  = spaces.Discrete(self.n_actions)
        obs_dim            = self.start_state.size * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 5) 其他参数
        self.max_steps  = int(max_steps)
        self.angle_tol  = float(angle_tol)

        # 6) runtime 状态
        self.state:     np.ndarray | None = None
        self._steps     = 0
        self._fig = self._ax = None   # for render()

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ):
        super().reset(seed=seed)      # 维护内部随机数
        self.state   = self.start_state.copy()
        self._steps  = 0
        info = {}
        return self._get_obs(), info

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."

        cur = self.state.copy()                       # 保证 contiguous
        next_state = self.dynamics.get_next_states(cur)[action][-1]
        self.state = next_state

        # —— reward：距离减少量 ——
        prev_d = np.linalg.norm(cur[:2]  - self.goal_state[:2])
        new_d  = np.linalg.norm(next_state[:2] - self.goal_state[:2])
        reward = prev_d - new_d

        # —— 终止 / 截断判定 ——
        self._steps += 1
        reached_pos = new_d < 1.0
        angle_err   = abs(np.arctan2(
            np.sin(next_state[2]-self.goal_state[2]),
            np.cos(next_state[2]-self.goal_state[2])
        ))
        reached_ang = angle_err < self.angle_tol
        terminated  = reached_pos and reached_ang
        truncated   = self._steps >= self.max_steps

        info: dict = {}
        return self._get_obs(), reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # 支持 MaskablePPO 的动作掩码
    # --------------------------------------------------------------------- #
    def action_masks(self) -> np.ndarray:
        """返回 (n_actions,) bool，True 表示可选。"""
        cur = self.state.copy()
        paths = self.dynamics.get_next_states(cur)[:, :, :3].copy()  # C-contiguous
        masks = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            masks[i] = self.world_cc.isValid(self.footprint, paths[i]).all()
        return masks

    # --------------------------------------------------------------------- #
    # Render: human / rgb_array
    # --------------------------------------------------------------------- #
    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode {mode}")

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()

        # 地图
        addGridToPlot(self.world_cc, ax)
        # 起点 / 终点
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        # 当前
        cur_xytheta = self.dynamics.get_xytheta_from_state(self.state)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, cur_xytheta)

        ax.set_title(f"Step {self._steps}")
        ax.set_aspect("equal")

        if mode == "human":
            plt.pause(0.001)
        else:   # rgb_array
            self._fig.canvas.draw_idle()
            # RGB array H×W×3
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = self._fig.canvas.get_width_height()
            return buf.reshape(h, w, 3)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = None

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal_state]).astype(np.float32)
