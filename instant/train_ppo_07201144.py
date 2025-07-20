#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 限制每个进程只用 1 线程，避免多线程抢核
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import yaml
import random
from pathlib import Path

import numpy as np
import torch
import imageio

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from domains_cc.map_and_scen_utils import parse_scen_file
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# ─── CustomEvalCallback: 单 env 串行评估 + GIF ─────────────────────────────────
class CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        make_env_fn,
        gif_dir: str,
        eval_freq: int,
        reward_threshold: float,
        n_eval_episodes: int = 5,
        fps: int = 10,
        prefix: str = "eval",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.eval_freq = eval_freq
        self.reward_threshold = reward_threshold
        self.n_eval_episodes = n_eval_episodes
        self.fps = fps
        self.prefix = prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            # 分别做随机与贪心评估
            results = {}
            for mode_name, det in [("stochastic", False), ("greedy", True)]:
                rews, lengths, succ = [], [], 0
                for _ in range(self.n_eval_episodes):
                    # 每条轨迹用一个新 env
                    wrapper = self.make_env_fn()
                    obs, _ = wrapper.reset()
                    total_r, steps, done, info = 0.0, 0, False, {}
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=det, action_masks=obs["action_mask"])
                        obs, r, done, _, info = wrapper.step(action)
                        total_r += r
                        steps += 1
                    rews.append(total_r)
                    lengths.append(steps)
                    if info.get("reached", False):
                        succ += 1
                mean_r = float(np.mean(rews))
                std_r  = float(np.std(rews))
                mean_l = float(np.mean(lengths))
                succ_rate = succ / self.n_eval_episodes
                results[mode_name] = mean_r
                print(
                    f"[Eval-{mode_name}] reward={mean_r:.2f}±{std_r:.2f} "
                    f"len={mean_l:.1f} succ={succ_rate:.0%}"
                )

            # greedy 达标则早停
            if results["greedy"] >= self.reward_threshold:
                print(f"[Eval] greedy mean_reward ≥ {self.reward_threshold}, stopping training.")
                return False

            # 生成 GIF：再次用一个 wrapper，贪心演示
            wrapper = self.make_env_fn()
            obs, _ = wrapper.reset()
            frames, done = [], False
            # unwrap 到原始 env 进行 render
            raw_env = wrapper.env
            while not done:
                action, _ = self.model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
                obs, _, done, _, _ = wrapper.step(action)
                frames.append(raw_env.render("rgb_array"))
            gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
            imageio.mimsave(str(gif_path), frames, fps=self.fps)
            print(f"[EvalAndGif] saved {gif_path}")
            self.count += 1

        return True

# ─── 环境工厂：支持覆盖 reward 参数 ─────────────────────────────────────────────
def make_maskable_env(
    scen_files, dynamics_config, footprint_config,
    time_cost=None, completion_bonus=None, seed=None
):
    def _init():
        random.seed(seed)
        scen = random.choice(scen_files)
        pairs = parse_scen_file(scen)
        idx = random.randrange(len(pairs))
        env = PathPlanningMaskEnv(
            scen_file=scen,
            problem_index=idx,
            dynamics_config=dynamics_config,
            footprint_config=footprint_config
        )
        # 覆盖 reward 参数
        if time_cost is not None:
            env.time_cost = time_cost
        if completion_bonus is not None:
            env.completion_bonus = completion_bonus
        if seed is not None:
            env.reset(seed=seed)
        return ActionMasker(env, lambda e: e.action_masks())
    return _init

# ─── 主流程 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ppo_config.yaml 路径")
    parser.add_argument("--logdir", default="logs", help="输出目录")
    args = parser.parse_args()

    cfg               = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed              = int(cfg.get("seed", 0))
    scen_files        = cfg["scen_files"]
    dynamics_config   = cfg["dynamics_config"]
    footprint_config  = cfg["footprint_config"]

    total_timesteps   = int(cfg.get("total_timesteps", 2_000_000))
    n_envs            = int(cfg.get("n_envs", 8))
    eval_freq         = int(cfg.get("eval_freq", 200_000))
    n_eval_episodes   = int(cfg.get("n_eval_episodes", 5))
    checkpoint_freq   = int(cfg.get("checkpoint_freq", eval_freq))
    reward_threshold  = float(cfg.get("reward_threshold", 100.0))

    # reward shaping
    time_cost         = float(cfg.get("time_cost",       -0.1))
    completion_bonus  = float(cfg.get("completion_bonus", 50.0))

    # PPO 超参
    policy            = cfg.get("policy", "MultiInputPolicy")
    policy_kwargs     = cfg.get("policy_kwargs", {})
    lr_cfg            = cfg.get("learning_rate", "linear")
    if isinstance(lr_cfg, str) and lr_cfg.lower() == "linear":
        init_lr        = float(cfg.get("init_learning_rate", 3e-4))
        learning_rate  = lambda p: init_lr * p
    else:
        learning_rate  = float(lr_cfg)

    n_steps           = int(cfg.get("n_steps",     4096))
    batch_size        = int(cfg.get("batch_size",  512))
    n_epochs          = int(cfg.get("n_epochs",    10))
    gamma             = float(cfg.get("gamma",     0.99))
    clip_range        = float(cfg.get("clip_range", 0.2))
    ent_coef          = float(cfg.get("ent_coef",   0.01))
    max_grad_norm     = float(cfg.get("max_grad_norm", 0.5))

    # prepare dirs
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    for d in (logdir, tb_dir, best_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Train env: parallel + monitor (raw reward) + normalize obs
    train_env = SubprocVecEnv([
        make_maskable_env(
            scen_files, dynamics_config, footprint_config,
            time_cost=time_cost,
            completion_bonus=completion_bonus,
            seed=seed + i
        ) for i in range(n_envs)
    ], start_method="fork")
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan','goal_vec','dist_grad','dist_phi'],
        clip_obs=10.0
    )

    # 2) Callbacks
    make_env_fn = make_maskable_env(
        scen_files, dynamics_config, footprint_config,
        time_cost=time_cost,
        completion_bonus=completion_bonus
    )
    eval_cb = CustomEvalCallback(
        make_env_fn=make_env_fn,
        gif_dir=str(best_dir / "gifs"),
        eval_freq=eval_freq,
        reward_threshold=reward_threshold,
        n_eval_episodes=n_eval_episodes,
        fps=10,
        prefix="eval",
        verbose=1
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(best_dir),
        name_prefix="ppo_ckpt"
    )
    callback = CallbackList([eval_cb, checkpoint_cb])

    # 3) Build & train
    model = MaskablePPO(
        policy=policy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        tensorboard_log=str(tb_dir),
        device="auto",
        verbose=1,
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted, closing environments…")
    finally:
        train_env.close()

    # 4) Save
    model.save(str(logdir / "final_model"))
    train_env.save(str(logdir / "vecnormalize.pkl"))
    print("✅ Training complete. Logs at", args.logdir)


if __name__ == "__main__":
    main()
