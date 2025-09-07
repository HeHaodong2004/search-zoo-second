from typing import Optional, List, Union, Tuple
import time
from dataclasses import dataclass, field
from collections import deque

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

from algorithms.conflictBasedSearch import AbstractIndividualSolver
from domains_cc.worldCC import WorldCollisionChecker, createFootprint
from domains_cc.worldCC_CBS import register_solver, WorldIndividualSolver, XYThetaPlusTimeSolution, WorldConstraint
from sb3_contrib import MaskablePPO
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot, plotPath, plotMultiPaths
from domains_cc.map_and_scen_utils import generate_random_map
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ！！！注意：
# 1) 默认使用 coverage runner 环境：GridWorld-v3-coverage-runner
# 2) 若该 Gym id 尚未注册，代码会自动 fallback 到直接导入类实例化
#    (domains_cc.path_planning_rl.env.planning_env_coverage_runner.PathPlanningCoverageEnvRunner)

def _r(x: float, p: int = 3) -> float:
    return float(np.round(x, p))

def _constraint_signature(constraints: Optional[List[WorldConstraint]]) -> Tuple:
    if not constraints:
        return tuple()
    sig = []
    for c in constraints:
        px, py = c.get_point()
        t = c.get_start_constraint_time()
        sig.append((_r(px), _r(py), _r(t)))
    return tuple(sig)

def _batch_obs(obs: dict) -> dict:
    return {k: np.expand_dims(v, 0) for k, v in obs.items()}

@dataclass
class CachedRollout:
    signature: Tuple
    path: np.ndarray
    times: np.ndarray
    actions: List[int]
    hist_seq: List[List[int]]
    frontier: deque = field(default_factory=deque)

@register_solver("RLSingleAgentSolver")
class RLSingleAgentSolver(WorldIndividualSolver):
    """
    Coverage 版单智能体求解器（保留原约束逻辑）：
    - 默认环境：GridWorld-v3-coverage-runner
    - 若未注册该 id，将回退为直接类构造
    - 通过 env_kwargs 透传 coverage 参数（如 cover_target_ratio 等）
    """
    def __init__(self, model_path: str, size: tuple[int, int],
                 max_iterations=100, debug_level=0, time_cutoff: int = 10,
                 replan_backoff_steps: int = 25,
                 replan_backoff_time: float = 0.0,
                 env_id: str = "GridWorld-v3-coverage-runner",
                 env_kwargs: Optional[dict] = None):
        super().__init__(time_cutoff, debug_level)
        self.model_path = model_path
        self.max_iterations = max_iterations
        self._xythetas: np.ndarray = np.array([[]])
        self._world_cc = None
        self._start_state = None
        self._goal_state = None
        self._footprint = None
        self.size = size
        self.visited_state_actions = set()
        self.solution = None

        self._base_env = None
        self._vec_env: Optional[DummyVecEnv] = None
        self._venv_norm: Optional[VecNormalize] = None
        self._model: Optional[MaskablePPO] = None
        self._last_rollout: Optional[CachedRollout] = None
        self._hist_len = 6
        # 复用原先的 vecnormalize 路径（观测结构一致）
        self._vecnorm_path = "domains_cc/path_planning_rl/logs/vecnormalize.pkl"

        self.replan_backoff_steps = int(replan_backoff_steps)
        self.replan_backoff_time = float(replan_backoff_time)

        # 新增：环境 ID 与覆盖参数
        self.env_id = str(env_id)
        # 给一些合理的 coverage 默认参数（可由 YAML 覆盖）
        self.env_kwargs = {
            # coverage 常用参数（可按需从 solver YAML 覆盖）
            "cover_target_ratio": 0.95,
            "cover_radius_m": 2.0,
            "subgoal_reach_tol_m": 1.0,
            "subgoal_bonus": 5.0,
            "sensor_range_m": None,
            "show_k_next": 20,
            "tour_cell_m": 2.0,
            "tour_start_from_row": "nearest",
            "tour_pick": "center",
            "order_strategy": "nn2opt",
            "cover_n_rays": 180,
            "goal_step_budget": 120,
            "no_improve_patience": 25,
            "min_progress_m": 0.10,
            "collision_patience": 10,
        }
        if env_kwargs:
            self.env_kwargs.update(env_kwargs)

    def get_name(self):
        return "RL"

    def set_start_goal_footprint(self, world_cc, start_state, goal_state, footprint, dynamics):
        """
        保持接口与上层一致；内部构造 coverage-runner 环境。
        """
        self._world_cc = world_cc
        self._start_state = start_state
        self._goal_state = goal_state
        self._footprint = footprint
        self.dynamics = dynamics

        # 仅首次构造环境与 VecNormalize / Model
        if self._base_env is None:
            base_make_kwargs = dict(
                problem_index=0,
                dynamics_config="domains_cc/dynamics/unicycle.yaml",
                footprint_config="domains_cc/footprints/footprint_car.yaml",
                max_steps=500,
                angle_tol=0.1,
                n_lasers=16,
                max_scan_dist=10.0,
                hist_len=6,
                start_xytheta=self._start_state,
                goal_xytheta=self._goal_state,   # goal 仅用于可视化/初始化
                world_cc=self._world_cc,
                constraints=[],
            )
            base_make_kwargs.update(self.env_kwargs)

            # 优先用 Gym id
            try:
                self._base_env = gym.make(self.env_id, **base_make_kwargs)
            except Exception as e:
                # 若尚未注册 id，fallback 到直接导入
                try:
                    from domains_cc.path_planning_rl.env.planning_env_coverage_runner import PathPlanningCoverageEnvRunner
                except Exception as e2:
                    raise RuntimeError(
                        f"无法创建 coverage 环境。Gym id={self.env_id} 未注册且无法导入类。"
                        f"\nGym error: {e}\nImport error: {e2}"
                    )
                self._base_env = PathPlanningCoverageEnvRunner(**base_make_kwargs)

            # 记录 hist_len；包装 VecEnv + VecNormalize
            self._hist_len = self._base_env.unwrapped.hist_len
            self._vec_env = DummyVecEnv([lambda: self._base_env])
            self._venv_norm = VecNormalize.load(self._vecnorm_path, self._vec_env)
            self._venv_norm.training = False
            self._venv_norm.norm_reward = False

        if self._model is None:
            self._model = MaskablePPO.load(self.model_path, env=self._venv_norm)

    @staticmethod
    def test_mask_fn(env):
        # 兼容旧调试接口
        return env.unwrapped.get_action_mask()

    # ----------- 保留原有“约束检查 + 动作屏蔽”逻辑 -----------
    def violates_any_constraint(self, start_xytheta, end_xytheta, start_time,
                                constraints, dt_default) -> bool:
        if constraints:
            for constraint in constraints:
                if constraint.violated_constraint(
                    np.array([start_xytheta, end_xytheta]),
                    np.array([start_time, start_time + dt_default]),
                    self._footprint, self._world_cc
                ):
                    return True
        return False

    def get_action_mask_vectorized(self, vec_env, t: float, constraints: Optional[List[WorldConstraint]]) -> np.ndarray:
        env = vec_env.envs[0].unwrapped
        base_mask = env.action_masks()  # 地图/占据约束
        old_state = env.state
        full_mask = base_mask.copy()
        for action in range(env.n_actions):
            if not base_mask[action]:
                full_mask[action] = False
                continue
            # 借用 runner 的 move() 取动作终点
            if hasattr(env, "move"):
                nxt = env.move(np.array([old_state[0], old_state[1], old_state[2]]), action)
            else:
                # 兜底：用 dynamics 推一次
                nxt = env.dynamics.get_next_states(old_state)[action, -1, :3]
            if self.violates_any_constraint(
                [old_state[0], old_state[1], old_state[2]],
                [nxt[0], nxt[1], nxt[2]],
                t, constraints, env.dt_default
            ):
                full_mask[action] = False
        return full_mask

    def _obs_after_restore(self) -> dict:
        raw = self._base_env.unwrapped._get_obs()
        return self._venv_norm.normalize_obs(raw)

    def get_solution(self, constraints: Optional[List[WorldConstraint]]) -> Optional[XYThetaPlusTimeSolution]:
        """
        运行 coverage 环境一步步 roll-out，保留你们原先的 cache + 约束剪枝。
        """
        assert self._venv_norm is not None and self._vec_env is not None and self._model is not None
        env_norm = self._venv_norm
        vec_env = self._vec_env
        uenv = vec_env.envs[0].unwrapped
        # coverage 版 runner 保留 set_constraints（可 noop），这里仍调用以便未来拓展
        if hasattr(uenv, "set_constraints"):
            uenv.set_constraints(constraints if constraints is not None else [])

        sig_now = _constraint_signature(constraints)
        dt = uenv.dt_default

        # ----- 前缀复用（遇到新约束回退到约束前片段）-----
        use_prefix = False
        prefix_cut_idx = -1
        prefix_path = None
        prefix_times = None
        prefix_hist = None

        if self._last_rollout is not None:
            sig_prev = self._last_rollout.signature
            if len(sig_now) >= len(sig_prev) and tuple(sig_now[:len(sig_prev)]) == sig_prev and len(sig_now) > 0:
                _, _, new_c_t = sig_now[-1]
                times_prev = self._last_rollout.times
                backoff = self.replan_backoff_steps * dt + self.replan_backoff_time
                cut_time = new_c_t - max(dt, backoff)
                idxs = [i for i, tt in enumerate(times_prev) if tt < cut_time]
                if len(idxs) > 0:
                    prefix_cut_idx = max(idxs)
                    use_prefix = True
                    prefix_path = self._last_rollout.path[:prefix_cut_idx + 1]
                    prefix_times = self._last_rollout.times[:prefix_cut_idx + 1]
                    prefix_hist = self._last_rollout.hist_seq[prefix_cut_idx]
                    print(f"[CACHE] replan from step {prefix_cut_idx} (t={times_prev[prefix_cut_idx]:.3f}), "
                          f"constraint @ {new_c_t:.3f}, backoff={backoff:.3f}s")

        # reset & 可选 restore
        obs = env_norm.reset()
        if use_prefix:
            _ = uenv.restore(state=prefix_path[-1], steps=prefix_cut_idx + 1, action_history=prefix_hist)
            obs = _batch_obs(self._obs_after_restore())

        path: List[np.ndarray] = []
        times: List[float] = []
        actions: List[int] = []
        hist_seq: List[List[int]] = []

        if use_prefix:
            for p, t in zip(prefix_path, prefix_times):
                path.append(np.asarray(p, dtype=float))
                times.append(float(t))
                hist_seq.append(list(prefix_hist))

        t_now = uenv._steps * uenv.dt_default
        terminated = False

        for _ in range(self.max_iterations):
            mask = self.get_action_mask_vectorized(vec_env, t_now, constraints).reshape(1, -1)
            action, _ = self._model.predict(obs, deterministic=True, action_masks=mask)

            hist_now = list(uenv.action_history)
            hist_seq.append(hist_now)

            obs, reward, done, info = env_norm.step(action)
            next_state = uenv.state

            path.append(next_state.astype(float))
            times.append(t_now)
            self._xythetas = np.array(path)

            if done[0]:
                terminated = True
                break

            t_now = uenv._steps * uenv.dt_default

        path_np = np.array(path)
        times_np = np.array(times)

        self._last_rollout = CachedRollout(
            signature=sig_now,
            path=path_np.copy(),
            times=times_np.copy(),
            actions=actions.copy(),
            hist_seq=[list(h) for h in hist_seq],
            frontier=deque()
        )

        if terminated and len(path_np) > 0:
            solution = XYThetaPlusTimeSolution(path_np, times_np, times_np[-1] if len(times_np) > 0 else 0.0)
            self.solution = solution
            np.random.seed(int(time.time() * 1e6) % 2**32)
            return solution
        else:
            # 没到终止（覆盖率不足/步数打断）时，依旧返回 None 以触发上层加约束，
            # 但同时把当前轨迹缓存到 self.solution 便于可视化
            self.solution = XYThetaPlusTimeSolution(
                path_np, times_np, times_np[-1] if len(times_np) > 0 else 0.0
            )
            np.random.seed(int(time.time() * 1e6) % 2**32)
            return None

    # ---------- 可视化（保持原接口） ----------
    def visualize(self, input_ax: Union[matplotlib.axes.Axes, str]) -> None:
        self.plotRL(input_ax, True)

    def plotRL(self, input_ax: Union[matplotlib.axes.Axes, str], shouldPlotFootprint: bool) -> None:
        if isinstance(input_ax, str):
            fig, ax = plt.subplots()
        else:
            ax = input_ax
        addGridToPlot(self._world_cc, ax)

        if self.solution is None:
            ax.set_title("RL - No Solution")
            if input_ax == "show":
                plt.show()
            elif isinstance(input_ax, str):
                plt.savefig(input_ax, dpi=300)
                plt.close()
            return

        xythetas = self.solution.get_xythetas()
        if shouldPlotFootprint:
            for pos in xythetas:
                addXYThetaToPlot(self._world_cc, ax, self._footprint, pos, computeValidity=True)
        else:
            ax.scatter(xythetas[:, 0], xythetas[:, 1], c='blue', s=1, zorder=2, alpha=0.5)

        xs = xythetas[:, 0]; ys = xythetas[:, 1]
        ax.plot(xs, ys, marker='o', linestyle='-', color='blue', alpha=0.7)
        ax.set_aspect('equal'); ax.set_title("RL Solution Path"); ax.grid(True)

        if input_ax == "show":
            plt.show()
        elif isinstance(input_ax, str):
            plt.savefig(input_ax, dpi=300)
            plt.close()

    def get_additional_stats_headers(self) -> List[str]:
        return []

    def get_additional_stats_values(self) -> List[float]:
        return []

# ---------- 本地快速测试（可选） ----------
def test_world_rl():
    original_map = generate_random_map(size=(20, 20), obstacle_prob=0.1)
    world_cc = WorldCollisionChecker(grid=original_map, resolution=1)
    footprint = createFootprint('rectangle', {'width': 0.5, 'height': 1.5, 'resolution': 1})
    world_rl = RLSingleAgentSolver(
        model_path="models_gridworld_single/best/best_model.zip",
        max_iterations=100,
        size=(20, 20),
        env_id="GridWorld-v3-coverage-runner",
        env_kwargs={"cover_target_ratio": 0.3, "max_steps": 300}
    )
    start_state = np.array([1.0, 1.0, 0.0])
    goal_state = np.array([18.0, 18.0, 0.0])

    world_rl.set_start_goal_footprint(world_cc, start_state, goal_state, footprint, None)
    solution = world_rl.get_solution(None)
    print("Solution:", solution)

    fig, ax = plt.subplots()
    addXYThetaToPlot(world_cc, ax, footprint, start_state, computeValidity=True)
    ax.text(start_state[0], start_state[1], 'Start', ha='center', va='bottom')
    addXYThetaToPlot(world_cc, ax, footprint, goal_state, computeValidity=True)
    ax.text(goal_state[0], goal_state[1], 'Goal', ha='center', va='bottom')

    if solution is not None:
        world_rl.plotRL(ax, True)
        xythetas = solution.get_xythetas()
        times = solution.get_times()
    else:
        xythetas = world_rl._xythetas
        times = np.arange(len(xythetas))
        if len(xythetas) > 0:
            ax.scatter(xythetas[:, 0], xythetas[:, 1], c='green', s=1, zorder=2, alpha=0.5)
            ax.plot(xythetas[:, 0], xythetas[:, 1], marker='o', linestyle='--', color='orange')
        else:
            ax.set_title("No steps taken")

    plt.show()

    if len(xythetas) > 0:
        plotPath(world_cc, "domains_cc/sophia/results/rl_solution.png", footprint, xythetas, times, computeValidity=True)
        plotMultiPaths(
            world_cc=world_cc,
            footprints=[footprint],
            xythetaplusses=[xythetas],
            times=[times],
            output_path="domains_cc/sophia/results/rl_solution.gif",
            computeValidity=True
        )
        print("Saved gif to domains_cc/sophia/results/rl_solution.gif")
    else:
        print("Nothing to visualize: no steps taken.")

if __name__ == "__main__":
    test_world_rl()
