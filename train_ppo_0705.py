# domains_cc/path_planning_rl/train/train_ppo.py
# -*- coding: utf-8 -*-
import os, argparse, yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from domains_cc.path_planning_rl.env.planning_env import PathPlanningMaskEnv


# -------------------------------------------------------------------- #
# Helper: build env  +  ActionMasker
# -------------------------------------------------------------------- #
def make_env(scen, idx, dyn_cfg, fp_cfg):
    env = PathPlanningMaskEnv(
        scen_file=scen,
        problem_index=idx,
        dynamics_config=dyn_cfg,
        footprint_config=fp_cfg,
    )
    # 新旧版本 ActionMasker 兼容
    try:
        return ActionMasker(env, lambda e: e.action_masks())          # sb3-contrib >= 2.2.1
    except TypeError:
        return ActionMasker(env, mask_fn=lambda e: e.action_masks())  # < 2.2.1


# -------------------------------------------------------------------- #
def main():
    # ------------------ CLI ------------------
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", required=True, help="ppo_config.yaml (UTF-8)")
    argp.add_argument("--logdir",  default="logs", help="输出目录")
    args = argp.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ------------------ cfg ------------------
    scen         = cfg["scen_file"]
    dyn_cfg      = cfg["dynamics_config"]
    fp_cfg       = cfg["footprint_config"]

    idx          = int(cfg.get("problem_index", 0))
    total_steps  = int(cfg.get("total_timesteps", 100_000))
    eval_freq    = int(cfg.get("eval_freq", 10_000))
    reward_stop  = float(cfg.get("reward_threshold", 1.0))

    # ------------------ IO ------------------
    os.makedirs(args.logdir, exist_ok=True)
    tb_log   = os.path.join(args.logdir, "tb")
    best_dir = os.path.join(args.logdir, "best_model")

    # ------------------ env ------------------
    train_env = make_env(scen, idx, dyn_cfg, fp_cfg)
    eval_env  = make_env(scen, idx, dyn_cfg, fp_cfg)

    # ------------------ callbacks ------------------
    stop_cb  = StopTrainingOnRewardThreshold(reward_threshold=reward_stop, verbose=1)
    eval_cb  = EvalCallback(eval_env, callback_after_eval=stop_cb,
                            eval_freq=eval_freq, best_model_save_path=best_dir,
                            verbose=1, render=False)

    # ------------------ PPO ------------------
    model = MaskablePPO(
        cfg.get("policy", "MlpPolicy"),
        train_env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps      =cfg.get("n_steps",      2048),
        batch_size   =cfg.get("batch_size",    64),
        n_epochs     =cfg.get("n_epochs",      10),
        gamma        =cfg.get("gamma",       0.99),
        clip_range   =cfg.get("clip_range",   0.2),
        ent_coef     =cfg.get("ent_coef",     0.0),
        tensorboard_log=tb_log,
        verbose=1,
    )

    # ------------------ train ------------------
    model.learn(total_timesteps=total_steps, callback=eval_cb)
    model.save(os.path.join(args.logdir, "final_model"))
    print(f"✅ 训练完成，模型与日志位于 {args.logdir}")


if __name__ == "__main__":
    main()
