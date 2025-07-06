# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
import os
import argparse
import yaml
import imageio
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    BaseCallback, CallbackList
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# —— 关键改动：引入 MaskableDummyVecEnv —— 
try:
    # sb3-contrib ≥2.3
    from sb3_contrib.common.maskable.vec_env import MaskableDummyVecEnv
except ImportError:
    # 回退：没有就当普通 DummyVecEnv 用（只能单 env）
    from stable_baselines3.common.vec_env import DummyVecEnv as MaskableDummyVecEnv

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

def make_maskable_env(scen, idx, dyn_cfg, fp_cfg):
    """返回一个已包装 ActionMasker(env) 的单智能体环境。"""
    env = PathPlanningMaskEnv(
        scen_file=scen,
        problem_index=idx,
        dynamics_config=dyn_cfg,
        footprint_config=fp_cfg,
    )
    try:
        # sb3-contrib ≥2.4
        return ActionMasker(env, mask_fn=lambda e: e.action_masks())
    except TypeError:
        # 旧版 sb3-contrib
        return ActionMasker(env,         lambda e: e.action_masks())

class GifCallback(BaseCallback):
    def __init__(self, scen, idx, dyn_cfg, fp_cfg, gif_dir,
                 prefix="eval_rollout", fps=10, verbose=0):
        super().__init__(verbose)
        self.gif_dir = Path(gif_dir); self.gif_dir.mkdir(exist_ok=True, parents=True)
        self.prefix, self.fps, self.count = prefix, fps, 0
        # demo_env（带 mask）用于预测，raw_env 只负责绘图
        self.demo_env = make_maskable_env(scen, idx, dyn_cfg, fp_cfg)
        self.raw_env  = PathPlanningMaskEnv(
            scen_file=scen, problem_index=idx,
            dynamics_config=dyn_cfg, footprint_config=fp_cfg
        )

    def _on_step(self) -> bool:
        obs, _ = self.demo_env.reset(); self.raw_env.reset()
        frames, done = [], False
        while not done:
            # 调用 predict 时 MaskableDummyVecEnv + ActionMasker 会把 mask 传进去
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.demo_env.step(action)
            self.raw_env.step(action)
            frames.append(self.raw_env.render(mode="rgb_array"))
            done = term or trunc

        path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
        imageio.mimsave(path, frames, fps=self.fps)
        if self.verbose:
            print(f"[GifCallback] saved {path}")
        self.count += 1
        return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="ppo_config.yaml")
    p.add_argument("--logdir",  default="logs", help="输出目录")
    args = p.parse_args()

    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scen       = cfg["scen_file"]
    dyn_cfg    = cfg["dynamics_config"]
    fp_cfg     = cfg["footprint_config"]
    idx        = int(cfg.get("problem_index", 0))
    total_ts   = int(cfg.get("total_timesteps", 100_000))
    eval_freq  = int(cfg.get("eval_freq", 10))
    reward_th  = float(cfg.get("reward_threshold", 1.0))

    os.makedirs(args.logdir, exist_ok=True)
    tb_dir   = os.path.join(args.logdir, "tensorboard")
    best_dir = os.path.join(args.logdir, "best_model")
    gif_dir  = os.path.join(args.logdir, "gifs")

    # ───── Vectorized Env ─────
    def _make(): return make_maskable_env(scen, idx, dyn_cfg, fp_cfg)
    # 这里用 MaskableDummyVecEnv，而不是普通的 DummyVecEnv
    train_env = VecMonitor(MaskableDummyVecEnv([_make]))
    eval_env  = VecMonitor(MaskableDummyVecEnv([_make]))

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_th, verbose=1)
    gif_cb  = GifCallback(scen, idx, dyn_cfg, fp_cfg, gif_dir, fps=10, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_after_eval=CallbackList([stop_cb, gif_cb]),
        eval_freq=eval_freq,
        best_model_save_path=best_dir,
        verbose=1,
        render=False,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps     =cfg.get("n_steps",    2048),
        batch_size  =cfg.get("batch_size",   64),
        n_epochs    =cfg.get("n_epochs",    10),
        gamma       =cfg.get("gamma",      0.99),
        clip_range  =cfg.get("clip_range", 0.2),
        ent_coef    =cfg.get("ent_coef",   0.0),
        tensorboard_log=tb_dir,
        verbose=1,
    )

    model.learn(total_timesteps=total_ts, callback=eval_cb)
    model.save(os.path.join(args.logdir, "final_model"))
    print("✅ training done, assets in", args.logdir)

if __name__ == "__main__":
    main()
