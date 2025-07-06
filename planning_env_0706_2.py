# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Dict

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot

class PathPlanningMaskEnv(gym.Env):
    """带动作掩码、角度判定和 render 的单机器人路径规划环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scen_file: str,
        problem_index: int,
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 500,
        angle_tol: float = 0.10,       # rad ≈ 5.7°
    ):
        # 1) 地图加载
        map_path = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_path)
        self.world_cc = WorldCollisionChecker(grid, resolution)

        # 2) 起终点加载 (x, y, θ)
        sg_pairs = parse_scen_file(scen_file)
        self.start_xytheta, self.goal_xytheta = sg_pairs[problem_index]

        # 3) 动力学 & footprint
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # 4) Action / Observation 空间
        self.n_actions    = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        obs_dim           = self.start_state.size * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 5) 其他参数
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)

        # 6) 运行时状态
        self.state:    Optional[np.ndarray] = None
        self._steps:   int = 0
        self._fig = None
        self._ax  = None

    '''def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ):
        super().reset(seed=seed)
        self.state  = self.start_state.copy()
        self._steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info'''
    
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.state  = self.start_state.copy()
        self._steps = 0
        obs = self._get_obs()
        info = {}

        # --- 新增：验证初始动作掩码中至少有一个合法动作 ---
        masks0 = self.action_masks()
        if not masks0.any():
            raise RuntimeError(
                f"Reset at {self.start_xytheta}: all {self.n_actions} primitives are invalid!"
            )
        return obs, info


    def step(self, action: int):
        assert self.state is not None, "请先调用 reset()"
        cur = self.state.copy()
        #next_state = self.dynamics.get_next_states(cur)[action, -1]
        #self.state = next_state

        traj = self.dynamics.get_next_states(cur)[action]   # 整段轨迹
        xyz_traj = traj[:, :3].copy(order="C")
        if not self.world_cc.isValid(self.footprint, xyz_traj).all():
            # 碰撞：给大负奖励并终止
            reward = -10.0
            terminated = truncated = True
            obs = self._get_obs()
            info = {"collision": True}
            return obs, reward, terminated, truncated, info

        next_state = traj[-1]
        self.state = next_state


        # reward = 距离减少量
        prev_d = np.linalg.norm(cur[:2] - self.goal_state[:2])
        new_d  = np.linalg.norm(next_state[:2] - self.goal_state[:2])
        reward = prev_d - new_d

        # 终止 / 截断判定
        self._steps += 1
        reached_pos = new_d < 1.0
        angle_err = abs(np.arctan2(
            np.sin(next_state[2] - self.goal_state[2]),
            np.cos(next_state[2] - self.goal_state[2])
        ))
        reached_ang = angle_err < self.angle_tol

        terminated = reached_pos and reached_ang
        truncated  = self._steps >= self.max_steps

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info


    def action_masks(self) -> np.ndarray:
        """MaskablePPO 所需：返回 (n_actions,) 布尔，True 表示可选。"""
        assert self.state is not None
        cur   = self.state.copy()
        paths = self.dynamics.get_next_states(cur)[:, :, :3].copy()  # C-contiguous
        masks = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            masks[i] = self.world_cc.isValid(self.footprint, paths[i]).all()
        return masks

    def render(self, mode='human'):
        """支持 human (plt.pause) 和 rgb_array (返回 H×W×3 数组)。"""
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax
        ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.goal_xytheta)
        cur_xytheta = self.state[:3]
        addXYThetaToPlot(self.world_cc, ax, self.footprint, cur_xytheta)
        ax.set_title(f"Step {self._steps}")

        if mode == 'rgb_array':
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            plt.pause(1 / self.metadata.get("render_fps", 10))

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = self._ax = None

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal_state]).astype(np.float32)
