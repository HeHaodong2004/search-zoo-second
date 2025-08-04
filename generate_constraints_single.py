#!/usr/bin/env python3
# generate_constraints.py

import numpy as np
import yaml
import csv
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from sb3_contrib import MaskablePPO

from domains_cc.map_and_scen_utils import parse_scen_file
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# ———— 根据实际情况修改下面常量 ————
CONFIG_PATH    = "domains_cc/path_planning_rl/configs/ppo_config.yaml"
MODEL_PATH     = "results/ppo_rectangle_unicycle/final_model.zip"
VECNORM_PATH   = "results/ppo_rectangle_unicycle/vecnormalize.pkl"
CONSTRAINT_CSV = "constraints.csv"


def make_env():
    """构造单环境，不包装 ActionMasker，保留完整 Obs Space。"""
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    scen = np.random.choice(cfg["scen_files"])
    pairs = parse_scen_file(scen)
    idx = np.random.randint(len(pairs))

    return PathPlanningMaskEnv(
        scen_file        = scen,
        problem_index    = idx,
        dynamics_config  = cfg["dynamics_config"],
        footprint_config = cfg["footprint_config"],
        max_steps        = cfg.get("max_steps", 1000)
    )


def main():
    # 1) 构造向量环境
    dummy    = DummyVecEnv([make_env])
    # 2) 加载归一化参数
    env_norm = VecNormalize.load(VECNORM_PATH, dummy)
    env_norm.training    = False
    env_norm.norm_reward = False
    # 3) 再挂上 Monitor
    env = VecMonitor(env_norm)
    # 4) 加载 MaskablePPO
    model = MaskablePPO.load(MODEL_PATH, env=env, device="auto")

    # 5) reset
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # 6) 拿到最底层的 PathPlanningMaskEnv
    # VecMonitor.venv -> VecNormalize -> DummyVecEnv.envs[0]
    pp_env = env.venv.envs[0]

    # 7) rollout 收集轨迹 (x,y) + 时间 t
    dt    = pp_env.dynamics.motion_primitives[0, -1]
    traj  = []
    times = []
    done  = False
    t     = 0.0

    while not done:
        # 手动从底层环境取 mask，并扩 batch dim
        mask = pp_env.action_masks()          # shape (n_actions,)
        mask = mask.reshape(1, -1)           # shape (1, n_actions)

        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=mask
        )
        step_out = env.step(action)

        # 兼容 Gymnasium (5-tuple) / SB3 (4-tuple)
        if len(step_out) == 5:
            obs, rews, terminated, truncated, infos = step_out
            dones = np.logical_or(terminated, truncated)
        else:
            obs, rews, dones, infos = step_out

        # 记录位置和时间
        x, y, _ = pp_env.state[:3]
        traj.append((float(x), float(y)))
        times.append(t)
        t += dt

        # 处理 dones
        if isinstance(dones, (list, np.ndarray)):
            done = bool(dones[0])
        else:
            done = bool(dones)

    # 8) 从轨迹上均匀采 N 个点作为时空约束
    N = 3
    if len(traj) > N+1:
        idxs = np.linspace(1, len(traj)-2, N, dtype=int)
    else:
        idxs = np.arange(1, len(traj)-1, dtype=int)
    constraints = [(traj[i][0], traj[i][1], times[i]) for i in idxs]

    # 9) 写入 CSV
    Path(CONSTRAINT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(CONSTRAINT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        for x, y, tv in constraints:
            writer.writerow([f"{x:.3f}", f"{y:.3f}", f"{tv:.3f}"])

    print(f"Saved {len(constraints)} constraints → {CONSTRAINT_CSV}")


if __name__ == "__main__":
    main()
