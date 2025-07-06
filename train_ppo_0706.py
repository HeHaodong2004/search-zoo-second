# -*- coding: utf-8 -*-
import os
import argparse
import yaml
import imageio
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

def make_maskable_env(scen, idx, dyn_cfg, fp_cfg):
    """构造一个带 action mask 的单智能体环境"""
    env = PathPlanningMaskEnv(
        scen_file=scen,
        problem_index=idx,
        dynamics_config=dyn_cfg,
        footprint_config=fp_cfg,
    )
    try:
        # 新版 sb3-contrib ≥2.2.1
        return ActionMasker(env, lambda e: e.action_masks())
    except TypeError:
        # 旧版 sb3-contrib <2.2.1
        return ActionMasker(env, mask_fn=lambda e: e.action_masks())

class GifCallback(BaseCallback):
    """
    每次 EvalCallback 完成评估后，分别在 demo_env 和 raw_env 上 reset，
    用 demo_env 计算 action_mask 和 next action，用 raw_env 来渲染保存 GIF。
    """
    def __init__(self,
                 scen: str,
                 idx: int,
                 dyn_cfg: str,
                 fp_cfg: str,
                 gif_dir: str,
                 prefix: str = "eval_rollout",
                 fps: int = 10,
                 verbose: int = 0):
        super().__init__(verbose)
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.fps = fps
        self.count = 0

        # 一个用来算 mask + 动作的包装 env
        self.demo_env = make_maskable_env(scen, idx, dyn_cfg, fp_cfg)
        # 一个干净 env 只用于 render
        self.raw_env  = PathPlanningMaskEnv(
            scen_file=scen,
            problem_index=idx,
            dynamics_config=dyn_cfg,
            footprint_config=fp_cfg,
        )

    def _on_step(self) -> bool:
        # 1) 同时 reset demo_env 和 raw_env
        demo_obs, _ = self.demo_env.reset()
        raw_obs, _  = self.raw_env.reset()

        frames = []
        done = False
        while not done:
            # 2) 从 demo_env 拿 mask，再预测 action
            mask = self.demo_env.action_masks()
            action, _ = self.model.predict(demo_obs, deterministic=True, action_masks=mask)

            # 3) 步 demo_env（更新内部 state，以便下一步 mask 正确）
            demo_obs, _, term1, trunc1, _ = self.demo_env.step(action)
            # 4) 同步步 raw_env 用于渲染
            raw_obs, _, term2, trunc2, _ = self.raw_env.step(action)

            # 5) render raw_env，收集帧
            frame = self.raw_env.render(mode='rgb_array')
            frames.append(frame)

            done = bool(term1 or trunc1)

        # 6) 存 GIF
        gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
        imageio.mimsave(str(gif_path), frames, fps=self.fps)
        if self.verbose:
            print(f"[GifCallback] Saved {gif_path}")
        self.count += 1
        return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="ppo_config.yaml (UTF-8 编码)")
    p.add_argument("--logdir", default="logs", help="输出目录")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scen       = cfg["scen_file"]
    dyn_cfg    = cfg["dynamics_config"]
    fp_cfg     = cfg["footprint_config"]
    idx        = int(cfg.get("problem_index", 0))
    total_steps= int(cfg.get("total_timesteps", 100_000))
    eval_freq  = int(cfg.get("eval_freq", 10))
    reward_thr = float(cfg.get("reward_threshold", 1.0))

    os.makedirs(args.logdir, exist_ok=True)
    tb_log_dir     = os.path.join(args.logdir, "tensorboard")
    best_model_dir = os.path.join(args.logdir, "best_model")
    gif_dir        = os.path.join(args.logdir, "gifs")

    # 构造并监控 VecEnv
    def _make(): return make_maskable_env(scen, idx, dyn_cfg, fp_cfg)
    train_env = VecMonitor(DummyVecEnv([_make]))
    eval_env  = VecMonitor(DummyVecEnv([_make]))

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_thr, verbose=1)
    gif_cb  = GifCallback(scen, idx, dyn_cfg, fp_cfg, gif_dir=gif_dir, prefix="eval_rollout", fps=10, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_after_eval=CallbackList([stop_cb, gif_cb]),
        eval_freq=eval_freq,
        best_model_save_path=best_model_dir,
        verbose=1,
        render=False,
    )

    model = MaskablePPO(
        policy=cfg.get("policy", "MlpPolicy"),
        env=train_env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps    =cfg.get("n_steps",     2048),
        batch_size =cfg.get("batch_size",    64),
        n_epochs   =cfg.get("n_epochs",      10),
        gamma      =cfg.get("gamma",       0.99),
        clip_range =cfg.get("clip_range",   0.2),
        ent_coef   =cfg.get("ent_coef",     0.0),
        tensorboard_log=tb_log_dir,
        verbose=1,
    )

    model.learn(total_timesteps=total_steps, callback=eval_cb)
    model.save(os.path.join(args.logdir, "final_model"))

    print("✅ Training finished.")
    print(f"• logs+model : {args.logdir}")
    print(f"• eval GIFs  : {gif_dir}")

if __name__ == "__main__":
    main()

