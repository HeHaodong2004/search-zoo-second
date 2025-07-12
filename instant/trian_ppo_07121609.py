# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
import os
import argparse
import yaml
from pathlib import Path
import imageio

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv

class EvalAndGifCallback(EvalCallback):
    """
    Extend EvalCallback: after each eval, build fresh masked envs,
    run deterministic rollout for GIF, and compute true success rate.
    """
    def __init__(
        self,
        env_kwargs: dict,
        gif_dir: str,
        fps: int = 10,
        prefix: str = "eval",
        **eval_kwargs
    ):
        super().__init__(**eval_kwargs)
        self.env_kwargs = env_kwargs
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.prefix = prefix
        self.count = 0
        # EvalCallback 已经传入 n_eval_episodes
        self.n_eval_episodes = eval_kwargs.get("n_eval_episodes", 1)

    def _on_step(self) -> bool:
        cont = super()._on_step()
        # 每 eval_freq 触发一次
        if self.n_calls % self.eval_freq == 0:
            # 1) 用全新 env 生成一条 GIF
            gif_env = ActionMasker(
                PathPlanningMaskEnv(**self.env_kwargs),
                lambda e: e.action_masks()
            )
            obs, _ = gif_env.reset()
            frames = []
            done = False
            while not done:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    action_masks=obs["action_mask"],
                )
                obs, _, done, _, _ = gif_env.step(action)
                frames.append(gif_env.env.render(mode="rgb_array"))
            out = self.gif_dir / f"{self.prefix}_{self.count:03d}.gif"
            imageio.mimsave(out, frames, fps=self.fps)
            print(f"[EvalAndGif] saved {out}")
            self.count += 1

            # 2) 用 n_eval_episodes 个全新 env 计算「真正」的成功率
            succ = 0
            for _ in range(self.n_eval_episodes):
                eval_env = ActionMasker(
                    PathPlanningMaskEnv(**self.env_kwargs),
                    lambda e: e.action_masks()
                )
                obs, _ = eval_env.reset()
                done = False
                info = {}
                while not done:
                    action, _ = self.model.predict(
                        obs,
                        deterministic=True,
                        action_masks=obs["action_mask"],
                    )
                    obs, _, done, _, info = eval_env.step(action)
                if info.get("reached", False):
                    succ += 1
            rate = succ / self.n_eval_episodes
            print(f"[Eval] success rate: {succ}/{self.n_eval_episodes} = {rate:.2%}")

        return cont

def make_maskable_env(scen_file, problem_index, dynamics_config, footprint_config):
    base = PathPlanningMaskEnv(
        scen_file=scen_file,
        problem_index=problem_index,
        dynamics_config=dynamics_config,
        footprint_config=footprint_config,
    )
    return ActionMasker(base, lambda e: e.action_masks())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ppo_config.yaml path")
    parser.add_argument("--logdir", default="logs", help="output dir")
    args = parser.parse_args()

    # load config
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    env_kwargs = dict(
        scen_file       = cfg["scen_file"],
        problem_index   = int(cfg.get("problem_index", 0)),
        dynamics_config = cfg["dynamics_config"],
        footprint_config= cfg["footprint_config"],
    )
    total_timesteps  = int(cfg.get("total_timesteps", 100_000))
    eval_freq        = int(cfg.get("eval_freq",     10_000))
    reward_threshold = float(cfg.get("reward_threshold",   1000.0))

    # prepare dirs
    logdir   = Path(args.logdir)
    tb_dir   = logdir / "tensorboard"
    best_dir = logdir / "best_model"
    gif_dir  = logdir / "gifs"
    for d in (logdir, tb_dir, best_dir, gif_dir):
        d.mkdir(parents=True, exist_ok=True)

    # vectorize + maskable
    def _make(): return make_maskable_env(**env_kwargs)
    train_env = VecMonitor(DummyVecEnv([_make]))
    eval_env  = VecMonitor(DummyVecEnv([_make]))

    # callbacks
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
    eval_cb = EvalAndGifCallback(
        env_kwargs             = env_kwargs,
        gif_dir                = str(gif_dir),
        prefix                 = "eval",
        fps                    = 10,
        eval_env               = eval_env,
        best_model_save_path   = str(best_dir),
        callback_on_new_best   = stop_cb,
        eval_freq              = eval_freq,
        n_eval_episodes        = 5,    # 比如跑 5 次评估
        deterministic          = True,
        render                 = False,
        verbose                = 1,
    )

    # train
    model = MaskablePPO(
        policy            = "MultiInputPolicy",
        env               = train_env,
        learning_rate     = cfg.get("learning_rate", 3e-4),
        n_steps           = cfg.get("n_steps",    2048),
        batch_size        = cfg.get("batch_size",   64),
        n_epochs          = cfg.get("n_epochs",     10),
        gamma             = cfg.get("gamma",      0.99),
        clip_range        = cfg.get("clip_range",  0.2),
        ent_coef          = cfg.get("ent_coef",    0.0),
        tensorboard_log   = str(tb_dir),
        verbose           = 1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(str(logdir / "final_model"))
    print("✅ Training complete. Outputs saved in", args.logdir)

if __name__ == "__main__":
    main()
