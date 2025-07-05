# domains_cc/path_planning_rl/env/planning_env.py

import numpy as np
import gym
from gym import spaces

from domains_cc.map_and_scen_utils import parse_map_file, parse_scen_file
from domains_cc.benchmark.parallel_benchmark_base import get_map_file_path
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path
from domains_cc.worldCC import WorldCollisionChecker

class PathPlanningMaskEnv(gym.Env):
    """
    Gym 环境：状态是 [state, goal_state] 串起来，动作是 motion_primitives 的索引。
    """

    def __init__(self,
                 scen_file: str,
                 problem_index: int,
                 dynamics_config: str,
                 footprint_config: str):
        # --- 1) 加载地图 & 场景 ---
        map_file = get_map_file_path(scen_file)
        grid, resolution = parse_map_file(map_file)
        self.world_cc = WorldCollisionChecker(grid, resolution)

        # --- 2) 加载起／终点 ---
        start_goal_pairs = parse_scen_file(scen_file)  # shape=(N,2,3)
        sg = start_goal_pairs[problem_index]
        self.start_xytheta = sg[0]
        self.goal_xytheta  = sg[1]

        # --- 3) 加载动力学 & 足迹 ---
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)

        # 内部状态：用 dynamics 转换
        self.start_state = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state  = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # --- 4) 定义 action / observation 空间 ---
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)

        state_dim = self.start_state.shape[0]
        obs_dim   = state_dim * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        self.state = None

    def reset(self):
        self.state = self.start_state.copy()
        return self._get_obs()

    def step(self, action: int):
        # 调用 dynamics，务必 copy 保证 contiguous
        cur = self.state.copy()
        traj = self.dynamics.get_next_states(cur.copy())[action]  # (steps+1, state_dim)
        next_state = traj[-1]
        self.state = next_state

        prev_d = np.linalg.norm(cur[:2]  - self.goal_state[:2])
        new_d  = np.linalg.norm(next_state[:2] - self.goal_state[:2])
        reward = prev_d - new_d
        done = bool(new_d < 1.0)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.state, self.goal_state]).astype(np.float32)

    def action_masks(self):
        """
        返回布尔 mask，False 表示该动作在当前 state 下会碰撞，需要被屏蔽。
        """
        cur = self.state.copy()
        all_trajs = self.dynamics.get_next_states(cur.copy())  # (N, steps+1, state_dim)

        mask = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            path = all_trajs[i, :, :3].copy()  # 保证 contiguous
            mask[i] = self.world_cc.isValid(self.footprint, path).all()
        return mask
