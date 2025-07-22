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
import imageio
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from domains_cc.map_and_scen_utils import parse_scen_file
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# ─── RandomResetEnv: 每次 reset 随机选场景 ────────────────────────────────────
class RandomResetEnv(gym.Env):
    def __init__(self,
                 scen_files,
                 dynamics_config,
                 footprint_config,
                 time_cost=None,
                 completion_bonus=None,
                 seed=None,
                 **env_kwargs):
        super().__init__()
        self.scen_files        = scen_files
        self.dynamics_config   = dynamics_config
        self.footprint_config  = footprint_config
        self._time_cost        = time_cost
        self._completion_bonus = completion_bonus
        self.env_kwargs        = env_kwargs

        # 只在初始化时 seed 一次，避免 eval 全部卡在同一场景
        if seed is not None:
            random.seed(seed)

        self._make_env()
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

    def _make_env(self):
        scen = random.choice(self.scen_files)
        pairs = parse_scen_file(scen)
        idx   = random.randrange(len(pairs))
        base = PathPlanningMaskEnv(
            scen_file       = scen,
            problem_index   = idx,
            dynamics_config = self.dynamics_config,
            footprint_config= self.footprint_config,
            **self.env_kwargs
        )
        # 应用奖励改写
        if self._time_cost is not None:
            base.time_cost = self._time_cost
        if self._completion_bonus is not None:
            base.completion_bonus = self._completion_bonus
        self.env = base

    def reset(self, **kwargs):
        # 每次 reset 都随机换场景
        self._make_env()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def action_masks(self):
        return self.env.action_masks()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# ─── CustomEvalCallback: 并行 VecEnv 评估 + GIF 演示 ─────────────────────────────
class CustomEvalCallback(BaseCallback):
    def __init__(self,
                 eval_env,
                 make_gif_env_fn,
                 gif_dir: str,
                 eval_freq: int,
                 reward_threshold: float,
                 n_eval_episodes: int = 5,
                 fps: int = 10,
                 prefix: str = "eval",
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.make_gif_env_fn = make_gif_env_fn
        self.gif_dir         = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.count           = 0
        self.eval_freq       = eval_freq
        self.reward_threshold= reward_threshold
        self.n_eval_episodes= n_eval_episodes
        self.fps             = fps
        self.prefix          = prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        # 从单进程 demo_env 获取 max_steps，防止卡死
        demo_for_max = self.make_gif_env_fn()
        max_steps = demo_for_max.env.env.max_steps

        # 并行评估：随机 vs 贪心
        results = {}
        for mode_name, det in [("stochastic", False), ("greedy", True)]:
            rews, lengths, succ = [], [], 0
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.zeros((self.eval_env.num_envs,), dtype=bool)
                total_r, steps, info = 0.0, 0, {}
                while not done.all() and steps < max_steps:
                    action, _ = self.model.predict(
                        obs,
                        deterministic=det,
                        action_masks=obs["action_mask"]
                    )
                    obs, rews_vec, dones, infos = self.eval_env.step(action)
                    r    = float(rews_vec[0])
                    done = dones
                    info = infos[0]
                    total_r += r
                    steps   += 1
                rews.append(total_r)
                lengths.append(steps)
                if info.get("reached", False):
                    succ += 1

            mean_r    = float(np.mean(rews))
            std_r     = float(np.std(rews))
            mean_l    = float(np.mean(lengths))
            succ_rate = succ / self.n_eval_episodes
            results[mode_name] = mean_r
            print(f"[Eval-{mode_name}] reward={mean_r:.2f}±{std_r:.2f} "
                  f"len={mean_l:.1f} succ={succ_rate:.0%}")

        if results["greedy"] >= self.reward_threshold:
            print(f"[Eval] greedy mean_reward ≥ {self.reward_threshold}, stopping training.")
            return False

        # 录制 GIF：单进程演示
        demo_env = self.make_gif_env_fn()
        obs, _  = demo_env.reset()
        raw_env  = demo_env.env.env  # PathPlanningMaskEnv
        frames, done, steps = [], False, 0
        while not done and steps < max_steps:
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=obs["action_mask"]
            )
            obs, r, done, _, info = demo_env.step(action)
            frames.append(raw_env.render("rgb_array"))
            steps += 1

        gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
        imageio.mimsave(str(gif_path), frames, fps=self.fps)
        print(f"[EvalAndGif] saved {gif_path}")
        self.count += 1

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True, help="ppo_config.yaml 路径")
    parser.add_argument("--logdir",  default="logs",    help="输出目录")
    args = parser.parse_args()

    cfg              = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed             = int(cfg.get("seed", 0))
    scen_files       = cfg["scen_files"]
    dynamics_config  = cfg["dynamics_config"]
    footprint_config = cfg["footprint_config"]

    total_timesteps  = int(cfg.get("total_timesteps", 2_000_000))
    n_envs           = int(cfg.get("n_envs", 8))
    n_eval_envs      = int(cfg.get("n_eval_envs", 4))
    eval_freq        = int(cfg.get("eval_freq", 200_000))
    n_eval_episodes  = int(cfg.get("n_eval_episodes", 5))
    checkpoint_freq  = int(cfg.get("checkpoint_freq", eval_freq))
    reward_threshold = float(cfg.get("reward_threshold", 800.0))

    time_cost        = float(cfg.get("time_cost",       -0.1))
    completion_bonus = float(cfg.get("completion_bonus", 50.0))

    policy           = cfg.get("policy", "MultiInputPolicy")
    policy_kwargs    = cfg.get("policy_kwargs", {})
    lr_cfg           = cfg.get("learning_rate", "linear")
    if isinstance(lr_cfg, str) and lr_cfg.lower() == "linear":
        init_lr       = float(cfg.get("init_learning_rate", 3e-4))
        learning_rate = lambda p: init_lr * p
    else:
        learning_rate = float(lr_cfg)

    n_steps        = int(cfg.get("n_steps",    4096))
    batch_size     = int(cfg.get("batch_size", 512))
    n_epochs       = int(cfg.get("n_epochs",   10))
    gamma          = float(cfg.get("gamma",    0.99))
    clip_range     = float(cfg.get("clip_range",0.2))
    ent_coef       = float(cfg.get("ent_coef", 0.01))
    max_grad_norm  = float(cfg.get("max_grad_norm",0.5))

    # 输出目录
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    for d in (logdir, tb_dir, best_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 训练环境工厂：带 seed 保持可复现
    def make_train_env(rank):
        def _init():
            return ActionMasker(
                RandomResetEnv(
                    scen_files, dynamics_config, footprint_config,
                    time_cost=time_cost,
                    completion_bonus=completion_bonus,
                    seed=seed + rank
                ),
                lambda e: e.action_masks()
            )
        return _init

    train_env = SubprocVecEnv(
        [make_train_env(i) for i in range(n_envs)],
        start_method="fork"
    )
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan', 'goal_vec', 'dist_grad', 'dist_phi'],
        clip_obs=10.0
    )

    # 评估环境工厂：不带 seed，保证每次真正随机
    def make_eval_env(_rank):
        def _init():
            return ActionMasker(
                RandomResetEnv(
                    scen_files, dynamics_config, footprint_config,
                    time_cost=time_cost,
                    completion_bonus=completion_bonus
                ),
                lambda e: e.action_masks()
            )
        return _init

    eval_env = SubprocVecEnv(
        [make_eval_env(i) for i in range(n_eval_envs)],
        start_method="fork"
    )
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan', 'goal_vec', 'dist_grad', 'dist_phi'],
        clip_obs=10.0
    )
    # 共享归一化统计
    eval_env.obs_rms  = train_env.obs_rms
    eval_env.clip_obs = train_env.clip_obs
    eval_env.training = False

    # 回调：并行评估 + GIF
    gif_env_fn = lambda: ActionMasker(
        RandomResetEnv(
            scen_files, dynamics_config, footprint_config,
            time_cost=time_cost,
            completion_bonus=completion_bonus
        ),
        lambda e: e.action_masks()
    )
    eval_cb = CustomEvalCallback(
        eval_env=eval_env,
        make_gif_env_fn=gif_env_fn,
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

    # MaskablePPO & 训练
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
        print("\n⚠️ Training interrupted by user.")
    finally:
        train_env.close()
        eval_env.close()

    # 保存模型 & 归一化参数
    model.save(str(logdir / "final_model"))
    train_env.save(str(logdir / "vecnormalize.pkl"))
    print("✅ Training complete. Outputs in", args.logdir)


if __name__ == "__main__":
    main()
