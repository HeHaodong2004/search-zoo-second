# domains_cc/path_planning_rl/env/planning_env_coverage_runner.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")

from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path


class PathPlanningCoverageEnvRunner(gym.Env):
    """
    Coverage-as-tour（Runner 版）：
      - 外部注入 world_cc / start_xytheta（不读取 scen/map）
      - 分块取代表点并排序为一条“巡游”序列（order=nn2opt|snake）
      - 逐个作为 point-goal 子目标交给 PPO
      - 覆盖用射线投射（不可透墙）
      - “快速切换”：步数预算 / 无进展 / 连续碰撞 触发跳过当前子目标
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        # —— 与 runner 接口对齐：外部注入世界与起止位姿 —— 
        problem_index: int,            # 占位，与 rl_sas 接口兼容
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 4000,
        angle_tol: float = 0.10,
        n_lasers: int = 16,
        max_scan_dist: float = 10.0,
        hist_len: int = 6,
        start_xytheta: Optional[np.ndarray] = None,
        goal_xytheta: Optional[np.ndarray] = None,
        world_cc: Optional[WorldCollisionChecker] = None,
        constraints: list = None,

        # —— coverage 相关参数（给默认值，必要时可在 solver_config 里透传）——
        cover_radius_m: float = 2.0,
        cover_target_ratio: float = 0.95,
        subgoal_reach_tol_m: float = 1.0,
        subgoal_bonus: float = 5.0,
        sensor_range_m: Optional[float] = None,
        show_k_next: int = 20,
        tour_cell_m: float = 2.0,
        tour_start_from_row: str = "nearest",
        tour_pick: str = "center",
        order_strategy: str = "nn2opt",
        cover_n_rays: int = 180,
        goal_step_budget: int = 120,
        no_improve_patience: int = 25,
        min_progress_m: float = 0.10,
        collision_patience: int = 10,

        # 奖励/代价
        time_cost: float = -0.2,
        completion_bonus: float = 80.0,
        collision_penalty: float = -10.0,
        stuck_step_penalty: float = 0.6,
        stuck_cap: int = 5,
    ):
        super().__init__()
        assert world_cc is not None, "coverage-runner 需要外部传入 world_cc"
        assert start_xytheta is not None, "coverage-runner 需要传入 start_xytheta"

        # —— dynamics/footprint/world —— 
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)
        self.world_cc = world_cc
        raw_grid = self.world_cc.grid    # 注意：world_cc.grid 是 top-origin & grid[x,y]
        self.resolution = float(self.world_cc.resolution)

        # bottom-origin free mask（1=free, 0=obs）
        bottom_grid = np.flipud(raw_grid)
        self.free_mask = (~bottom_grid).astype(np.uint8)
        self.H, self.W = bottom_grid.shape

        # —— 起止位姿（goal 仅用于可视化/初始化，不是完成判据）——
        self.start_xytheta = np.array(start_xytheta, dtype=np.float32)
        self.goal_xytheta  = np.array(goal_xytheta if goal_xytheta is not None else start_xytheta, dtype=np.float32)
        self.start_state   = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state    = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # —— 覆盖/子目标/快速切换参数 —— 
        self.cover_radius_m = float(cover_radius_m)
        self.cover_target_ratio = float(cover_target_ratio)
        self.subgoal_reach_tol_m = float(subgoal_reach_tol_m)
        self.subgoal_bonus = float(subgoal_bonus)
        self.cover_n_rays = int(cover_n_rays)
        self.sensor_range_m = float(sensor_range_m) if sensor_range_m is not None else float(max_scan_dist)

        self.node_strategy = "tour"
        self.show_k_next = int(show_k_next)
        self.tour_cell_m = float(tour_cell_m)
        self.tour_start_from_row = str(tour_start_from_row)
        self.tour_pick = str(tour_pick)
        self.order_strategy = str(order_strategy)

        self.goal_step_budget = int(goal_step_budget)
        self.no_improve_patience = int(no_improve_patience)
        self.min_progress_m = float(min_progress_m)
        self.collision_patience = int(collision_patience)

        # —— 奖励相关 —— 
        self.time_cost = float(time_cost)
        self.completion_bonus = float(completion_bonus)
        self.collision_penalty = float(collision_penalty)
        self.stuck_step_penalty = float(stuck_step_penalty)
        self.stuck_cap = int(stuck_cap)

        # —— 动作/观测空间，与 P2P policy 完全一致 —— 
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        self.hist_len = int(hist_len)
        self.action_history = deque(maxlen=self.hist_len)

        self.n_lasers = int(n_lasers)
        self.max_scan_dist = float(max_scan_dist)
        self.laser_angles = np.linspace(-np.pi, np.pi, self.n_lasers, endpoint=False)

        self.observation_space = spaces.Dict({
            "scan":        spaces.Box(0.0, self.max_scan_dist, (self.n_lasers,), dtype=np.float32),
            "goal_vec":    spaces.Box(-np.inf, np.inf, (2,),       dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
            "hist":        spaces.MultiDiscrete([self.n_actions]*self.hist_len),
            "dist_grad":   spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "dist_phi":    spaces.Box(0.0, 1.0,   (1,), dtype=np.float32),
        })

        # —— 运行状态 —— 
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax  = None
        self.constraints = list(constraints) if constraints is not None else []

        # —— 内部势场/梯度/子目标/覆盖 —— 
        self.dist_map = None
        self.grad_x = None
        self.grad_y = None
        self.max_potential = 0.0
        self.collision_streak = 0
        self.subgoal_count = 0
        self._goal_reachable_from_robot = True

        self.high_nodes_seq: List[Tuple[int,int]] = []
        self._nodes_used = np.zeros((self.H, self.W), dtype=np.uint8)

        self._no_progress_steps = 0
        self._best_phi_this_goal = np.inf
        self._best_goal_dist = np.inf
        self._collision_steps = 0
        self._goal_steps = 0

        # 时间步长（供 rl_sas 使用）
        self.dt_default = float(self.dynamics.motion_primitives[0, -1])

        # —— 初始化：构建首个子目标与势场 —— 
        self._init_fields_and_tour()

    # ====== 与 rl_sas 兼容的“runner 接口” ======
    def set_constraints(self, constraints):
        # coverage 先不在环境层面处理约束（rl_sas 会在 action mask 中处理）
        self.constraints = list(constraints) if constraints is not None else []

    def move(self, old_state, action):
        traj = self.dynamics.get_next_states(old_state)[action, :, :3]
        return traj[-1]

    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        mask = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            if not self.world_cc.isValid(self.footprint, trajs[i]).all():
                continue
            mask[i] = True
        return mask

    # ====== reset/step/render ======
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        self.collision_streak = 0
        self.subgoal_count = 0
        self.covered = np.zeros((self.H, self.W), dtype=np.uint8)

        if options:
            if "cover_radius_m" in options:
                self.cover_radius_m = float(options["cover_radius_m"])
            if "sensor_range_m" in options:
                self.sensor_range_m = float(options["sensor_range_m"])
                self.max_scan_dist  = float(options["sensor_range_m"])

        self._mark_coverage(self.state[:2])
        self._build_high_level_nodes_tour()
        self._pick_first_reachable_goal()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()
        info = {
            "collision": not valid, "reached": False,
            "covered_ratio": 0.0, "subgoals": self.subgoal_count,
            "next_nodes_world": []
        }

        # 碰撞/正常推进 + 基础奖励
        if not valid:
            self.collision_streak += 1
            extra = - self.stuck_step_penalty * min(max(self.collision_streak - 1, 0), self.stuck_cap)
            reward = self.collision_penalty + self.time_cost + extra
            self.action_history.append(action)
        else:
            self.state = traj[-1]
            self.action_history.append(action)
            if self.collision_streak > 0:
                self.collision_streak = 0

            prev_phi = self._interpolate_dist(cur[:2])
            new_phi  = self._interpolate_dist(self.state[:2])
            delta_potential = prev_phi - new_phi
            reward = delta_potential + self.time_cost

            # 更新覆盖
            self._mark_coverage(self.state[:2])

        self._steps += 1
        self._goal_steps += 1

        # === 停滞统计（进展用“势能/欧氏距离改善”双准则） ===
        goal_dist = float(np.hypot(self.goal_state[0] - self.state[0],
                                   self.goal_state[1] - self.state[1]))
        cur_phi = float(self._interpolate_dist(self.state[:2]))
        if valid:
            phi_improved  = (self._best_phi_this_goal - cur_phi) > 1e-3
            dist_improved = (self._best_goal_dist     - goal_dist) > self.min_progress_m
            if phi_improved or dist_improved:
                self._no_progress_steps = 0
                self._best_phi_this_goal = min(self._best_phi_this_goal, cur_phi)
                self._best_goal_dist     = min(self._best_goal_dist, goal_dist)
            else:
                self._no_progress_steps += 1
            self._collision_steps = 0
        else:
            self._no_progress_steps += 1
            self._collision_steps += 1

        # === 快速切换触发：步数预算 / 无进展 / 连续碰撞 ===
        def _try_advance_goal():
            tried = 0
            while tried < 64:
                next_rc = self._pick_next_goal_rc()
                if next_rc is None:
                    return False
                self._set_goal_by_rc(*next_rc)
                rr, cc = self._world_to_grid_rc(self.state[:2])
                if self._goal_reachable_from_robot and np.isfinite(self.dist_map[rr, cc]):
                    self.subgoal_count += 1
                    info["subgoals"] = self.subgoal_count
                    return True
                tried += 1
            return False

        if (self._goal_steps >= self.goal_step_budget) or \
           (self._no_progress_steps >= self.no_improve_patience) or \
           (self._collision_steps >= self.collision_patience):
            _ = _try_advance_goal()
            # 重置统计
            self._no_progress_steps = 0
            self._best_phi_this_goal = np.inf
            self._best_goal_dist = np.inf
            self._collision_steps = 0
            self._goal_steps = 0

        # 正常到达判定
        dxg = self.state[0] - self.goal_state[0]
        dyg = self.state[1] - self.goal_state[1]
        reached = valid and (np.hypot(dxg, dyg) < self.subgoal_reach_tol_m)
        if reached:
            reward += self.subgoal_bonus
            info["reached"] = True
            _ = _try_advance_goal()
            self._no_progress_steps = 0
            self._best_phi_this_goal = np.inf
            self._best_goal_dist = np.inf
            self._collision_steps = 0
            self._goal_steps = 0

        # 覆盖率
        total_free = int(np.count_nonzero(self.free_mask == 1))
        covered_free = int(np.count_nonzero((self.free_mask == 1) & (self.covered == 1)))
        cover_ratio = covered_free / max(total_free, 1)
        info["covered_ratio"] = cover_ratio

        done = (cover_ratio >= self.cover_target_ratio)
        truncated = False
        if not done and self._steps >= self.max_steps:
            truncated = True
        if done:
            reward += self.completion_bonus

        # 未来K个节点（可视化）
        if self.high_nodes_seq:
            remain = []
            count = 0
            for (r, c) in self.high_nodes_seq:
                if count >= self.show_k_next:
                    break
                if self._nodes_used[r, c] == 1 or self.covered[r, c] == 1:
                    continue
                wx, wy = (c + 0.5)*self.resolution, (r + 0.5)*self.resolution
                remain.append([wx, wy]); count += 1
            info["next_nodes_world"] = remain

        return self._get_obs(), reward, done, truncated, info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax; ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])

        # 势场 & 覆盖热力
        if self.dist_map is not None:
            H, W = self.dist_map.shape
            dm = self.dist_map / (self.max_potential + 1e-6)
            ax.imshow(dm, origin='lower',
                      extent=[0, W*self.resolution, 0, H*self.resolution],
                      cmap='viridis', alpha=0.45)
        if hasattr(self, "covered"):
            cov = np.ma.masked_where(self.covered == 0, self.covered)
            ax.imshow(cov, origin='lower',
                      extent=[0, self.W*self.resolution, 0, self.H*self.resolution],
                      cmap='autumn', alpha=0.35)

        # 未来 K 个节点（可视化）
        if self.high_nodes_seq:
            xs, ys = [], []; plotted = 0
            for (r, c) in self.high_nodes_seq:
                if plotted >= self.show_k_next: break
                if self._nodes_used[r, c] == 1 or self.covered[r, c] == 1: continue
                wx, wy = (c + 0.5)*self.resolution, (r + 0.5)*self.resolution
                ax.plot(wx, wy, marker='o', markersize=3)
                ax.text(wx+0.05, wy+0.05, f"{plotted+1}", fontsize=7)
                xs.append(wx); ys.append(wy); plotted += 1
            if len(xs) >= 2: ax.plot(xs, ys, linewidth=1.0)

        ax.set_xlim(0, self.W*self.resolution); ax.set_ylim(0, self.H*self.resolution)
        ax.set_aspect('equal'); ax.set_title(f"Step {self._steps} | Subgoals {self.subgoal_count} | order={self.order_strategy}")

        if mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            import time as _t; _t.sleep(1/self.metadata["render_fps"])

    # ====== 初始化：构建巡游节点并选择首个可达子目标 ======
    def _init_fields_and_tour(self):
        # 先做一次覆盖和巡游节点构建，在 reset 时也会再做一遍
        self.covered = np.zeros((self.H, self.W), dtype=np.uint8)
        self._mark_coverage(self.start_xytheta[:2])
        self._build_high_level_nodes_tour()
        self._pick_first_reachable_goal()

    # ---------- helpers ----------
    def _world_to_grid_rc(self, xy: np.ndarray) -> Tuple[int, int]:
        r = int(np.floor(xy[1] / self.resolution))
        c = int(np.floor(xy[0] / self.resolution))
        r = np.clip(r, 0, self.H-1)
        c = np.clip(c, 0, self.W-1)
        return r, c

    def _mark_coverage(self, xy: np.ndarray):
        """可见性覆盖：射线投射，不透墙（在 bottom-origin free_mask 上）"""
        rad_cells = int(np.ceil(self.cover_radius_m / self.resolution))
        cx = float(xy[0] / self.resolution)  # 列方向
        cy = float(xy[1] / self.resolution)  # 行方向
        angles = np.linspace(0.0, 2*np.pi, self.cover_n_rays, endpoint=False)
        for ang in angles:
            dx = np.cos(ang); dy = np.sin(ang)
            for step in range(rad_cells + 1):
                gx = int(np.floor(cx + dx * step))
                gy = int(np.floor(cy + dy * step))
                if not (0 <= gy < self.H and 0 <= gx < self.W):
                    break
                if self.free_mask[gy, gx] == 0:
                    break
                self.covered[gy, gx] = 1

    # ---------- distance/gradient ----------
    def _compute_distance_map(self, grid: np.ndarray, goal_rc: np.ndarray) -> np.ndarray:
        """
        grid: 1=free, 0=obs ; goal_rc=(r,c)（注意 r 是行、c 是列）
        修正了索引顺序：必须用 [r,c] 访问，而不是 [c,r]
        """
        from collections import deque
        H, W = grid.shape
        free = (grid == 1)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        gr, gc = map(int, goal_rc)
        # 若起点在障碍上，找最近 free
        if not free[gr, gc]:
            visited = np.zeros_like(free, bool)
            dq = deque([(gr, gc)])
            visited[gr, gc] = True
            neighs = [(-1,0),(1,0),(0,-1),(0,1)]
            found = False
            while dq and not found:
                r0, c0 = dq.popleft()
                for dr, dc in neighs:
                    rn, cn = r0+dr, c0+dc
                    if 0<=rn<H and 0<=cn<W and not visited[rn,cn]:
                        if free[rn,cn]:
                            gr, gc = rn, cn
                            found = True
                            break
                        visited[rn,cn] = True
                        dq.append((rn,cn))

        dist[gr, gc] = 0.0
        dq = deque([(gr, gc)])
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        while dq:
            r0, c0 = dq.popleft()
            for dr, dc in neighs:
                rn, cn = r0+dr, c0+dc
                if (0<=rn<H and 0<=cn<W and free[rn,cn]
                        and not np.isfinite(dist[rn,cn])):
                    dist[rn,cn] = dist[r0,c0] + 1.0
                    dq.append((rn,cn))
        return dist

    def _build_grid_gradient(self, dist_map: np.ndarray, free_mask: np.ndarray):
        H, W = dist_map.shape
        gx = np.zeros_like(dist_map, dtype=np.float32)
        gy = np.zeros_like(dist_map, dtype=np.float32)
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in range(H):
            for j in range(W):
                if free_mask[i, j] == 0 or not np.isfinite(dist_map[i, j]):
                    continue
                best_d = dist_map[i, j]
                bi, bj = i, j
                for di, dj in neighs:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and free_mask[ni, nj] == 1 and np.isfinite(dist_map[ni, nj]):
                        if dist_map[ni, nj] < best_d:
                            best_d = dist_map[ni, nj]
                            bi, bj = ni, nj
                if (bi != i) or (bj != j):
                    vx = (bj - j); vy = (bi - i)
                    norm = (vx*vx + vy*vy) ** 0.5
                    if norm > 0:
                        gx[i, j] = vx / norm
                        gy[i, j] = vy / norm
        return gx, gy

    # ---------- tour 节点生成 + 排序 ----------
    def _tour_pick_in_cell(self, points: List[Tuple[int,int]], ri: int, cj: int, cell: int) -> Optional[Tuple[int,int]]:
        if not points:
            return None
        if self.tour_pick == "center":
            r0 = ri*cell; c0 = cj*cell
            r_center = r0 + min(cell, self.H - r0)/2.0
            c_center = c0 + min(cell, self.W - c0)/2.0
            best, best_d2 = None, 1e18
            for (r, c) in points:
                d2 = (r - r_center)**2 + (c - c_center)**2
                if d2 < best_d2:
                    best, best_d2 = (r, c), d2
            return best
        else:  # medoid
            pts = np.array(points, dtype=int)
            dsum = np.abs(pts[:,None,:] - pts[None,:,:]).sum(axis=2).sum(axis=1)
            idx = int(np.argmin(dsum))
            return tuple(map(int, pts[idx]))

    def _build_high_level_nodes_tour(self):
        cell = max(1, int(round(self.tour_cell_m / self.resolution)))
        Hc = int(np.ceil(self.H / cell))
        Wc = int(np.ceil(self.W / cell))

        sc_free: List[List[List[Tuple[int,int]]]] = [[[] for _ in range(Wc)] for __ in range(Hc)]
        for r in range(self.H):
            ri = r // cell
            for c in range(self.W):
                if self.free_mask[r, c] != 1:
                    continue
                cj = c // cell
                sc_free[ri][cj].append((r, c))

        buckets: List[List[List[Tuple[int,int]]]] = [[[] for _ in range(Wc)] for __ in range(Hc)]
        for ri in range(Hc):
            for cj in range(Wc):
                rc = self._tour_pick_in_cell(sc_free[ri][cj], ri, cj, cell)
                if rc is not None:
                    buckets[ri][cj].append(rc)

        # 起始行：nearest/top/bottom
        if self.tour_start_from_row == "nearest":
            start_r = int(np.clip(self.start_xytheta[1] / self.resolution, 0, self.H-1))
            start_ri = start_r // cell
        elif self.tour_start_from_row == "top":
            start_ri = Hc - 1
        else:
            start_ri = 0

        rows_up = list(range(start_ri, Hc, 1))
        rows_down = list(range(start_ri-1, -1, -1))
        row_order = rows_up + rows_down

        snake_seq: List[Tuple[int,int]] = []
        reverse = False
        for ri in row_order:
            cols = list(range(Wc))
            if reverse:
                cols.reverse()
            for cj in cols:
                if not buckets[ri][cj]:
                    continue
                if len(buckets[ri][cj]) > 1:
                    r0 = ri*cell; c0 = cj*cell
                    r_center = r0 + min(cell, self.H - r0)/2.0
                    c_center = c0 + min(cell, self.W - c0)/2.0
                    buckets[ri][cj].sort(key=lambda rc: (rc[0]-r_center)**2 + (rc[1]-c_center)**2)
                snake_seq.extend(buckets[ri][cj])
            reverse = not reverse

        def _dedup_valid(seq):
            uniq, seen = [], set()
            for (r, c) in seq:
                if not (0 <= r < self.H and 0 <= c < self.W):
                    continue
                if self.free_mask[r, c] != 1:
                    continue
                if (r, c) in seen:
                    continue
                uniq.append((r, c)); seen.add((r, c))
            return uniq

        snake_seq = _dedup_valid(snake_seq)

        if self.order_strategy == "snake" or len(snake_seq) <= 2:
            self.high_nodes_seq = snake_seq
            self._nodes_used[:] = 0
            return

        # nn2opt
        nodes = snake_seq[:]
        rs, cs = self._world_to_grid_rc(np.array(self.start_xytheta[:2]))

        # 起点附近的 geodesic 最近
        best_idx, best_d = -1, np.inf
        for i, (r, c) in enumerate(nodes):
            dist_map = self._compute_distance_map(self.free_mask, np.array([r, c], dtype=int))
            d = dist_map[rs, cs]
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx < 0:
            self.high_nodes_seq = snake_seq
            self._nodes_used[:] = 0
            return
        order = [nodes.pop(best_idx)]

        # geodesic 最近邻串起来
        while nodes:
            cur = order[-1]
            dist_map = self._compute_distance_map(self.free_mask, np.array([cur[0], cur[1]], dtype=int))
            best_i, best_d = -1, np.inf
            for i, (r, c) in enumerate(nodes):
                d = dist_map[r, c]
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i < 0:
                d2s = [ (order[-1][0]-r)**2 + (order[-1][1]-c)**2 for (r,c) in nodes ]
                best_i = int(np.argmin(d2s))
            order.append(nodes.pop(best_i))

        # 2-opt（欧氏）
        def _eu_len(a: Tuple[int,int], b: Tuple[int,int]) -> float:
            return np.hypot(a[0]-b[0], a[1]-b[1])

        def _two_opt(route: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
            N = len(route)
            if N < 4:
                return route
            for i in range(N-3):
                for k in range(i+2, N-1):
                    a, b = route[i], route[i+1]
                    c, d = route[k], route[k+1]
                    if _eu_len(a,c) + _eu_len(b,d) + 1e-9 < _eu_len(a,b) + _eu_len(c,d):
                        route[i+1:k+1] = reversed(route[i+1:k+1])
            return route

        order = _two_opt(order)
        self.high_nodes_seq = _dedup_valid(order)
        self._nodes_used[:] = 0

    # ---------- goal/fields ----------
    def _set_goal_by_rc(self, r: int, c: int):
        self.goal_xytheta = np.array([c*self.resolution, r*self.resolution, 0.0], dtype=np.float32)
        self.goal_state   = self.dynamics.get_state_from_xytheta(self.goal_xytheta)
        dist_arr = self._compute_distance_map(self.free_mask, np.array([r, c]))

        self._goal_reachable_from_robot = True
        if self.state is not None:
            rr, cc = self._world_to_grid_rc(self.state[:2])
            self._goal_reachable_from_robot = np.isfinite(dist_arr[rr, cc])

        unreachable = np.isinf(dist_arr)
        reachable   = ~unreachable
        self.max_potential = float(np.max(dist_arr[reachable])) if np.any(reachable) else 0.0
        dist_arr[unreachable] = self.max_potential
        self.dist_map = dist_arr

        self.grad_x, self.grad_y = self._build_grid_gradient(self.dist_map, self.free_mask)
        valid = (self.free_mask == 1) & (~unreachable)
        self.grad_x[~valid] = 0.0
        self.grad_y[~valid] = 0.0

        # reset per-goal stats
        self._no_progress_steps = 0
        self._best_phi_this_goal = np.inf
        self._best_goal_dist = np.inf
        self._collision_steps = 0
        self._goal_steps = 0

    def _pick_next_goal_rc(self) -> Optional[Tuple[int,int]]:
        if self.high_nodes_seq:
            for (r, c) in self.high_nodes_seq:
                if self._nodes_used[r, c] == 1:
                    continue
                if self.covered[r, c] == 1:
                    self._nodes_used[r, c] = 1
                    continue
                self._nodes_used[r, c] = 1
                return r, c
        return None

    def _pick_first_reachable_goal(self):
        # 预选“可达”的首目标（一次就跳过不可达/已覆盖的）
        while True:
            first_rc = self._pick_next_goal_rc()
            if first_rc is None:
                # 找不到就把当前位置作为临时目标，避免空场
                self._set_goal_by_rc(*self._world_to_grid_rc(self.start_xytheta[:2]))
                break
            r, c = first_rc
            dist_arr = self._compute_distance_map(self.free_mask, np.array([r, c]))
            rr, cc = self._world_to_grid_rc(self.start_xytheta[:2])
            if np.isfinite(dist_arr[rr, cc]):
                self._set_goal_by_rc(r, c)
                self.subgoal_count += 1
                break
            # 不可达则继续挑下一个（已在 _pick_next_goal_rc 标为 used）

    # ---------- observation ----------
    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]

        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        gw_x, gw_y = self.world_cc.grid.shape  # 注意：top-origin & grid[x,y]
        max_steps = int(self.sensor_range_m / self.resolution)
        for idx, ang in enumerate(self.laser_angles):
            dx = np.cos(theta+ang); dy = np.sin(theta+ang)
            for s in range(max_steps):
                px = x + dx*s*self.resolution
                py = y + dy*s*self.resolution
                ix = int(np.floor(px/self.resolution))
                iy = int(np.floor(py/self.resolution))
                if not (0 <= ix < gw_x and 0 <= iy < gw_y) or self.world_cc.grid[ix, iy] != 0:
                    scans[idx] = s * self.resolution
                    break

        dxg = self.goal_state[0]-x; dyg = self.goal_state[1]-y
        gx =  dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy =  dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        gx_w = self._interpolate_grad(self.grad_x, self.state[:2])
        gy_w = self._interpolate_grad(self.grad_y, self.state[:2])
        gx_r = gx_w*np.cos(-theta) - gy_w*np.sin(-theta)
        gy_r = gx_w*np.sin(-theta) + gy_w*np.cos(-theta)

        phi = self._interpolate_dist(self.state[:2])
        phi_norm = phi / (self.max_potential + 1e-6)

        hist = np.array(self.action_history, dtype=np.int64)
        return {
            "scan":        scans,
            "goal_vec":    goal_vec,
            "action_mask": self.action_masks(),
            "hist":        hist,
            "dist_grad":   np.array([gx_r, gy_r], dtype=np.float32),
            "dist_phi":    np.array([phi_norm], dtype=np.float32),
        }

    # ---------- interpolation ----------
    def _interpolate_dist(self, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.H-1)
        j_f = np.clip(x/self.resolution, 0, self.W-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.H-1), min(j0+1, self.W-1)
        di, dj = i_f - i0, j_f - j0
        d00 = self.dist_map[i0, j0]; d10 = self.dist_map[i1, j0]
        d01 = self.dist_map[i0, j1]; d11 = self.dist_map[i1, j1]
        return (
            d00*(1-di)*(1-dj)
          + d10*di*(1-dj)
          + d01*(1-di)*dj
          + d11*di*dj
        )

    def _interpolate_grad(self, grad_map: np.ndarray, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.H-1)
        j_f = np.clip(x/self.resolution, 0, self.W-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.H-1), min(j0+1, self.W-1)
        di, dj = i_f - i0, j_f - j0
        g00 = grad_map[i0, j0]; g10 = grad_map[i1, j0]
        g01 = grad_map[i0, j1]; g11 = grad_map[i1, j1]
        return (
            g00*(1-di)*(1-dj)
          + g10*di*(1-dj)
          + g01*(1-di)*dj
          + g11*di*dj
        )

    # ---------- runtime setters（可选） ----------
    def set_cover_radius(self, r_m: float):
        self.cover_radius_m = float(r_m)

    def set_sensor_range(self, r_m: float):
        self.sensor_range_m = float(r_m)
        self.max_scan_dist  = float(r_m)

    # ---------- export helper（可选） ----------
    def get_high_level_nodes_world(self):
        res = []
        for (r, c) in self.high_nodes_seq:
            x = (c + 0.5) * self.resolution
            y = (r + 0.5) * self.resolution
            res.append([float(x), float(y)])
        return res

    def export_nodes_json(self, path: str):
        import json
        nodes = self.get_high_level_nodes_world()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes_xy": nodes}, f, ensure_ascii=False, indent=2)
        return path
