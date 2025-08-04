#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
from pathlib import Path

import numpy as np
import imageio
import yaml
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import MaskablePPO

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv
from domains_cc.worldCC_CBS import WorldConstraint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",         required=True, help="ppo_config.yaml 路径")
    p.add_argument("--model",          required=True, help="策略 .zip 文件")
    p.add_argument("--vecnorm",        required=True, help="归一化参数 .pkl")
    p.add_argument("--constraint_dir", required=True, help="constraints_*.csv 所在目录")
    p.add_argument("--gif_dir",        default="test_gifs", help="GIF 输出目录")
    p.add_argument("--deterministic",  action="store_true", help="用贪心策略")
    args = p.parse_args()

    # 1) 读配置
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    scen_files = cfg["scen_files"]

    # 2) 加载策略模型（暂不绑定 env）
    model = MaskablePPO.load(args.model, device="auto")

    # 3) 输出目录
    gif_root = Path(args.gif_dir)
    gif_root.mkdir(parents=True, exist_ok=True)

    # 4) 遍历所有 CSV 约束文件，并按 scen_files 顺序＋文件名里 index 排序
    raw_csvs = list(Path(args.constraint_dir).glob("*.csv"))
    def csv_sort_key(p: Path):
        stem = p.stem                     # e.g. "maze-32-32-2@basic_03"
        scen_nm, idx = stem.rsplit("_", 1)
        # 在 cfg["scen_files"] 里找 scen_nm 的位置
        scen_i = len(scen_files)
        for i, s in enumerate(scen_files):
            stem_s = Path(s).stem
            if stem_s == scen_nm or scen_nm in stem_s:
                scen_i = i
                break
        return (scen_i, int(idx))
    csv_files = sorted(raw_csvs, key=csv_sort_key)

    # 5) 开始循环
    rewards, lengths, successes = [], [], 0

    for run_idx, csv_f in enumerate(csv_files):
        # 提取 scen 名和 index
        stem = csv_f.stem  # 例如 "Cantwell@basic_03"
        scen_name, idx_str = stem.rsplit("_", 1)
        problem_index = int(idx_str)

        # 查找对应 scen 文件路径
        scen_file = None
        for s in scen_files:
            stem_s = Path(s).stem
            if stem_s == scen_name or scen_name in stem_s:
                scen_file = s
                break
        if scen_file is None:
            raise ValueError(f"No scenario file matching '{scen_name}' among:\n" +
                             "\n".join(Path(s).stem for s in scen_files))

        # 读取约束
        constrs = []
        with open(csv_f) as f:
            for row in csv.reader(f):
                x, y, t = map(float, row)
                constrs.append(WorldConstraint(
                    ag=0,
                    point=np.array([x, y]),
                    time=t
                ))

        pp = PathPlanningMaskEnv(
            scen_file        = scen_file,
            problem_index    = problem_index,
            dynamics_config  = cfg["dynamics_config"],
            footprint_config = cfg["footprint_config"],
            max_steps        = cfg.get("max_steps", 1000),
        )

        # 补上奖励参数与约束
        pp.time_cost        = cfg.get("time_cost", -0.1)
        pp.completion_bonus = cfg.get("completion_bonus", 50.0)
        pp.set_constraints(constrs)          # ← 这里注入 CSV 里读出的约束

        # ② 再包进 DummyVecEnv（先有 pp，再写 lambda）
        dummy = DummyVecEnv([lambda pp=pp: pp])
        env = VecNormalize.load(args.vecnorm, dummy)
        env.training = False
        env.norm_reward = False
        env = VecMonitor(env)

        # Reset 环境
        obs = env.reset()

        done = False
        ep_r, steps = 0.0, 0
        frames = []

        while not done and steps < cfg.get("max_steps", 1000):
            # 生成 mask
            raw_mask = pp.action_masks()
            mask = raw_mask.reshape(1, -1)

            action, _ = model.predict(
                obs,
                deterministic=args.deterministic,
                action_masks=mask
            )
            obs, rews, dones, infos = env.step(action)
            r = float(rews[0])
            done = bool(dones[0])
            info = infos[0]

            ep_r += r
            steps += 1

            frames.append(pp.render("rgb_array"))

        # 保存 gif
        out_gif = gif_root / f"run_{run_idx:02d}.gif"
        imageio.mimsave(str(out_gif), frames, fps=10)
        print(f"[Run {run_idx:02d}] scen={scen_name}#{problem_index} "
              f"reward={ep_r:.1f} len={steps} → {out_gif}")

        rewards.append(ep_r)
        lengths.append(steps)
        if info.get("reached", False):
            successes += 1

    # 汇总
    N = len(csv_files)
    print("\n===== Summary =====")
    print(f"runs          : {N}")
    print(f"mean reward   : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"mean length   : {np.mean(lengths):.1f}")
    print(f"success rate  : {successes/N:.0%}")
    print(f"GIFs in       : {gif_root.resolve()}")
