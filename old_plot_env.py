    def plot_env(self, step):
        # 固定布局与像素尺寸，避免帧间抖动
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y', 'm', 'c']
        ncols = 1 + 2 * N_AGENTS
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), constrained_layout=False)

        # ---------------- Panel 1: Global belief + traj + current poses （不画 intent） ----------------
        ax_global = axes[0]
        ax_global.imshow(self.env.global_belief, cmap='gray', zorder=0)  # 置底
        ax_global.axis('off')

        # 机器人轨迹与当前位置
        for r in self.robots:
            c = color_list[r.id % len(color_list)]
            try:
                # 当前位置（world -> index）
                cell = get_cell_position_from_coords(r.location, r.map_info)
                ax_global.plot(cell[0], cell[1], c + 'o', markersize=8, zorder=6)

                # 全局轨迹（world -> index）
                if hasattr(r, 'trajectory_x') and len(r.trajectory_x) > 1:
                    xs = (np.array(r.trajectory_x) - r.map_info.map_origin_x) / r.cell_size
                    ys = (np.array(r.trajectory_y) - r.map_info.map_origin_y) / r.cell_size
                    ax_global.plot(xs, ys, c, linewidth=2, zorder=2)
            except Exception:
                pass
        ax_global.set_title('Global Belief + Traj + Poses')

        # —— 叠加 rendezvous 可视化（若有） —— #
        try:
            if getattr(self, 'debug_rdv', None) is not None and self.debug_rdv is not None:
                if isinstance(self.debug_rdv, tuple) and len(self.debug_rdv) >= 2 and self.debug_rdv[0] is not None:
                    center_xy, r_meet, meta = self.debug_rdv
                    # 世界 -> 索引
                    center_cell = get_cell_position_from_coords(center_xy, self.robots[0].map_info)

                    H, W = self.env.global_belief.shape
                    if 0 <= center_cell[0] < W and 0 <= center_cell[1] < H:
                        # RDV 中心点（更醒目：红点 + 白描边）
                        ax_global.plot(center_cell[0], center_cell[1],
                                    marker='o',
                                    markersize=9,
                                    markerfacecolor='red',
                                    markeredgecolor='white',
                                    markeredgewidth=1.5,
                                    linestyle='None',
                                    zorder=10)

                        # 画会合带（米 -> 像素）
                        import matplotlib.patches as patches
                        r_pix = max(1.0, float(r_meet) / float(self.robots[0].cell_size))
                        circ = patches.Circle((center_cell[0], center_cell[1]), r_pix,
                                            linewidth=1.6, fill=False, linestyle='--',
                                            edgecolor='red', alpha=0.95, zorder=9)
                        ax_global.add_patch(circ)

                        # 文本（Score/ETA）
                        s_total = float(meta.get('total', 0.0)) if isinstance(meta, dict) else 0.0
                        t_eta  = float(meta.get('lat_est', 0.0)) if isinstance(meta, dict) else 0.0
                        ax_global.text(center_cell[0] + 3, center_cell[1] + 3,
                                    f"RDV\nS={s_total:.2f}\nT={t_eta:.1f}m",
                                    fontsize=9, color='black', zorder=11,
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    else:
                        # 可选：调试输出（坐标越界）
                        # print(f"[plot] RDV out of bounds: cell={center_cell}, map={W}x{H}")
                        pass
        except Exception:
            pass

        # ------------- Panels: for each agent -> [Predicted (local) | Observation(local with known intents)] -------------
        col = 1
        for viewer, r in enumerate(self.robots):
            # ---- (1) Predicted local
            ax_pred = axes[col]; col += 1
            ax_pred.axis('off')
            try:
                if r.pred_max_map_info is not None:
                    pred_local = r.get_updating_map(r.location, base=r.pred_max_map_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)
                    ax_pred.imshow(pred_local.map, cmap='gray', vmin=0, vmax=255, zorder=0)
                    alpha_mask = (belief_local.map == FREE) * 0.5
                    ax_pred.imshow(belief_local.map, cmap='Blues', alpha=alpha_mask, zorder=1)
                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=10, zorder=5)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center')
                ax_pred.set_title(f'Agent {r.id} Predicted (local)')
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err: {e}', ha='center', va='center', fontsize=8)

            # ---- (2) Observation local（关键：intent 的起点对齐）
            ax_obs = axes[col]; col += 1
            ax_obs.axis('off')
            try:
                # 画当前 viewer 的局部图与当前位置
                obs_map = r.updating_map_info.map
                ax_obs.imshow(obs_map, cmap='gray', zorder=0)
                rc2 = get_cell_position_from_coords(r.location, r.updating_map_info)
                ax_obs.plot(rc2[0], rc2[1], 'mo', markersize=10, zorder=6, label='self')

                # 叠加 viewer 已知的队友位置
                if hasattr(self, 'last_known_locations'):
                    known_positions = self.last_known_locations[viewer]
                    for aid in range(N_AGENTS):
                        pos = known_positions[aid]
                        try:
                            pc = get_cell_position_from_coords(pos, r.updating_map_info)
                            ax_obs.plot(pc[0], pc[1],
                                        color_list[aid % len(color_list)] + 'o',
                                        markersize=6, zorder=5)
                        except Exception:
                            pass  # 不在窗口内或越界

                # 叠加 viewer 已知的 intents（对齐修正：拼接起点）
                if hasattr(self, 'last_known_intents'):
                    intents_view = self.last_known_intents[viewer]  # dict: {aid -> path(list[[x,y],...])}
                    for aid, path in intents_view.items():
                        if not path:
                            continue

                        # 将 intent 的世界坐标映射到 viewer 的 local updating map
                        # 并拼接起点：own -> 自己当前位置；others -> 该 agent 最后已知位置（若存在）
                        path_cells = []
                        try:
                            if aid == r.id:
                                start_cell = get_cell_position_from_coords(r.location, r.updating_map_info)
                                path_cells.append(start_cell)
                            else:
                                if hasattr(self, 'last_known_locations'):
                                    pos_aid = self.last_known_locations[viewer][aid]
                                    try:
                                        start_cell = get_cell_position_from_coords(pos_aid, r.updating_map_info)
                                        path_cells.append(start_cell)
                                    except Exception:
                                        pass  # 不在窗口内则无法拼接起点

                            # 映射后续 intent 目标点
                            for p in path:
                                try:
                                    cell = get_cell_position_from_coords(np.array(p, dtype=float), r.updating_map_info)
                                    path_cells.append(cell)
                                except Exception:
                                    continue  # 不在本地窗口内则跳过

                        except Exception:
                            path_cells = []

                        # 绘制
                        if len(path_cells) > 0:
                            ix = [c[0] for c in path_cells]
                            iy = [c[1] for c in path_cells]
                            style = ':' if aid != r.id else '--'
                            zord = 4 if aid != r.id else 5
                            colr = color_list[aid % len(color_list)]
                            ax_obs.plot(ix, iy,
                                        linestyle=style,
                                        marker='x',
                                        linewidth=1.2,
                                        markersize=5,
                                        color=colr,
                                        zorder=zord,
                                        label=f'i{aid}')

                ax_obs.set_title(f'Agent {r.id} Local Obs (+ known intents)')
                try:
                    ax_obs.legend(loc='lower right', fontsize='x-small')
                except Exception:
                    pass

            except Exception as e:
                ax_obs.text(0.5, 0.5, f'Obs plot err: {e}', ha='center', va='center', fontsize=8)

        # 不使用 tight_layout，保证画布尺寸稳定
        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        self.env.frame_files.append(out_path)
