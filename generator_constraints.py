#!/usr/bin/env python3
# generate_constraints.py

import argparse
import numpy as np
import yaml
import csv
from pathlib import Path

from sb3_contrib import MaskablePPO
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv
from domains_cc.map_and_scen_utils import parse_scen_file

# 1) 常量或 CLI 都行；我们这里走 CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="为一批 .scen 文件里的问题生成时空约束 CSV"
    )
    p.add_argument(
        "--config",
        required=True,
        help="ppo_config.yaml（包含 scen_files 或 scen_dir）"
    )
    p.add_argument(
        "--model",
        required=True,
        help="训练好的 MaskablePPO 模型 .zip"
    )
    p.add_argument(
        "--output_dir",
        default="constraints_sets",
        help="输出 CSV 的目录"
    )
    p.add_argument(
        "--per",
        type=int,
        default=10,
        help="每个问题采样多少个约束点"
    )
    return p.parse_args()


def get_scen_files(cfg):
    """
    从 config 里拿 scen_files 或者 scen_dir，
    返回一个按字母排序的场景文件路径列表
    """
    scen_input = cfg.get("scen_dir", None) or cfg.get("scen_files", [])
    if isinstance(scen_input, str) and Path(scen_input).is_dir():
        # 整个目录，查所有 .scen
        files = sorted(Path(scen_input).glob("*.scen"))
    else:
        # 直接就是一个列表
        files = scen_input
    return [str(p) for p in files]


def sample_constraints(env, model, N):
    """
    对单个 PathPlanningMaskEnv，跑一次 rollout，
    均匀抽 N 个 (x,y,t) 点作为约束
    """
    obs, _ = env.reset()
    done = False
    t = 0.0
    dt = env.dynamics.motion_primitives[0, -1]
    traj, times = [], []
    while not done:
        mask = env.action_masks().reshape(1, -1)
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=mask
        )
        obs, _, done, truncated, _ = env.step(action)
        done = done or truncated
        x, y, _ = env.state[:3]
        traj.append((x, y))
        times.append(t)
        t += dt

    L = len(traj)
    if L > N+1:
        idxs = np.linspace(1, L-2, N, dtype=int)
    else:
        idxs = np.arange(1, L-1, dtype=int)
    return [(traj[i][0], traj[i][1], times[i]) for i in idxs]


def main():
    args = parse_args()

    # 1) 读配置文件
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 2) 准备模型
    model = MaskablePPO.load(args.model)

    # 3) 找到所有场景文件，保持顺序
    scen_files = get_scen_files(cfg)

    # 4) 准备输出目录
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)

    # 5) 对每个场景，生成它里面每个子问题的约束 CSV
    for scen_path in scen_files:
        scen_name = Path(scen_path).stem
        pairs = parse_scen_file(scen_path)
        for idx in range(len(pairs)):
            # 构造环境
            env = PathPlanningMaskEnv(
                scen_file        = scen_path,
                problem_index    = idx,
                dynamics_config  = cfg["dynamics_config"],
                footprint_config = cfg["footprint_config"],
                max_steps        = cfg.get("max_steps", 1000)
            )
            # 采样约束
            constraints = sample_constraints(env, model, args.per)

            # 写成 CSV：<scene>_<idx>.csv
            out_file = outdir / f"{scen_name}_{idx:02d}.csv"
            with open(out_file, "w", newline="") as f:
                writer = csv.writer(f)
                for x,y,t in constraints:
                    writer.writerow([f"{x:.3f}", f"{y:.3f}", f"{t:.3f}"])
            print(f"Saved {len(constraints)} → {out_file}")

    print("All done!")


if __name__ == "__main__":
    main()
