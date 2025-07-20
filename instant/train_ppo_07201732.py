#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Limit each process to one thread to avoid BLAS/OMP contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import yaml
import random
from pathlib import Path

import numpy as np
import imageio
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from domains_cc.map_and_scen_utils import parse_scen_file
from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

# ─── RandomResetEnv: each reset picks a new random scenario ─────────────────────
class RandomResetEnv(gym.Env):
    def __init__(self, scen_files, dynamics_config, footprint_config,
                 time_cost=None, completion_bonus=None, **env_kwargs):
        super().__init__()
        self.scen_files       = scen_files
        self.dynamics_config  = dynamics_config
        self.footprint_config = footprint_config
        # store reward overrides
        self._time_cost        = time_cost
        self._completion_bonus = completion_bonus
        self.env_kwargs        = env_kwargs
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
        # apply reward overrides
        if self._time_cost is not None:
            base.time_cost = self._time_cost
        if self._completion_bonus is not None:
            base.completion_bonus = self._completion_bonus
        self.env = base

    def reset(self, **kwargs):
        self._make_env()
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def action_masks(self):
        return self.env.action_masks()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# ─── CustomEvalCallback: uses eval_env (with shared normalization) ────────────
class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, gif_dir, eval_freq, reward_threshold,
                 n_eval_episodes=5, fps=10, prefix="eval", verbose=1):
        super().__init__(verbose)
        self.eval_env         = eval_env
        self.gif_dir          = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.count            = 0
        self.eval_freq        = eval_freq
        self.reward_threshold = reward_threshold
        self.n_eval_episodes  = n_eval_episodes
        self.fps              = fps
        self.prefix           = prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        results = {}
        for mode_name, det in [("stochastic", False), ("greedy", True)]:
            rews, lengths, succ = [], [], 0
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.zeros((self.eval_env.num_envs,), dtype=bool)
                total_r, steps, info = 0.0, 0, {}
                while not done.all():
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

            mean_r   = float(np.mean(rews))
            std_r    = float(np.std(rews))
            mean_l   = float(np.mean(lengths))
            succ_rate= succ / self.n_eval_episodes
            results[mode_name] = mean_r
            print(f"[Eval-{mode_name}] reward={mean_r:.2f}±{std_r:.2f} "
                  f"len={mean_l:.1f} succ={succ_rate:.0%}")

        if results["greedy"] >= self.reward_threshold:
            print(f"[Eval] greedy mean_reward ≥ {self.reward_threshold}, stopping training.")
            return False

        # Generate GIF demonstration (first sub-env)
        obs = self.eval_env.reset()
        done = np.zeros((self.eval_env.num_envs,), dtype=bool)
        frames = []
        # Unwrap to the underlying PathPlanningMaskEnv:
        # eval_env -> VecNormalize -> VecMonitor -> DummyVecEnv -> ActionMasker -> RandomResetEnv -> PathPlanningMaskEnv
        action_masker = self.eval_env.venv.venv.envs[0]     # DummyVecEnv.envs[0]
        rand_reset    = action_masker.env                   # RandomResetEnv
        raw_env       = rand_reset.env                      # PathPlanningMaskEnv
        while not done.all():
            action, _ = self.model.predict(
                obs,
                deterministic=True,
                action_masks=obs["action_mask"]
            )
            obs, rews_vec, dones, infos = self.eval_env.step(action)
            done = dones
            frames.append(raw_env.render("rgb_array"))
        gif_path = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
        imageio.mimsave(str(gif_path), frames, fps=self.fps)
        print(f"[EvalAndGif] saved {gif_path}")
        self.count += 1

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ppo_config.yaml")
    parser.add_argument("--logdir", default="logs",    help="Output directory")
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
    reward_threshold = float(cfg.get("reward_threshold", 100.0))

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

    n_steps    = int(cfg.get("n_steps",    4096))
    batch_size = int(cfg.get("batch_size", 512))
    n_epochs   = int(cfg.get("n_epochs",   10))
    gamma      = float(cfg.get("gamma",    0.99))
    clip_range = float(cfg.get("clip_range",0.2))
    ent_coef   = float(cfg.get("ent_coef", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm",0.5))

    # Prepare output dirs
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    for d in (logdir, tb_dir, best_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Training env: parallel + monitor raw reward + normalize obs only
    train_env = SubprocVecEnv([
        lambda i=i: ActionMasker(
            RandomResetEnv(scen_files, dynamics_config, footprint_config,
                           time_cost=time_cost,
                           completion_bonus=completion_bonus),
            lambda e: e.action_masks()
        )
        for i in range(n_envs)
    ], start_method="fork")
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan','goal_vec','dist_grad','dist_phi'],
        clip_obs=10.0
    )

    # 2) Eval   env: parallel random-reset + share normalize stats
    eval_env = DummyVecEnv([
        lambda i=i: ActionMasker(
            RandomResetEnv(scen_files, dynamics_config, footprint_config,
                           time_cost=time_cost,
                           completion_bonus=completion_bonus),
            lambda e: e.action_masks()
        )
        for i in range(n_eval_envs)
    ])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True, norm_reward=False,
        norm_obs_keys=['scan','goal_vec','dist_grad','dist_phi'],
        clip_obs=10.0
    )
    # Share the running stats from train_env
    eval_env.obs_rms  = train_env.obs_rms
    eval_env.clip_obs = train_env.clip_obs
    eval_env.training = False

    # 3) Callbacks
    eval_cb = CustomEvalCallback(
        eval_env=eval_env,
        gif_dir=str(best_dir/"gifs"),
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

    # 4) Build and train
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

    # 5) Save final model & normalizer
    model.save(str(logdir/"final_model"))
    train_env.save(str(logdir/"vecnormalize.pkl"))
    print("✅ Training complete. Outputs in", args.logdir)


if __name__ == "__main__":
    main()
