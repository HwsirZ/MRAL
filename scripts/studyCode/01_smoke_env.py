import habitat
from omegaconf import OmegaConf
import random
import numpy as np

# 新增
import os
import imageio.v2 as imageio

CFG_PATH = "/data1/user2025263063/project/MARLObjectNav/configs/studyCode/pointnav.yaml"

# ====== GIF 配置 ======
GIF_DIR = "/data1/user2025263063/project/MARLObjectNav/gifs"
SAVE_GIF = True
FPS = 30                 # gif 播放帧率
RECORD_EVERY = 1        # 每隔多少 step 记录一帧（减小文件）
MAX_FRAMES = 1000        # 最多保存多少帧（防止爆内存/文件太大）
# =====================

def get_collision(info: dict) -> bool:
    if not isinstance(info, dict):
        return False
    if "collisions" in info and isinstance(info["collisions"], dict):
        return bool(info["collisions"].get("is_collision", False))
    if "is_collision" in info:
        return bool(info["is_collision"])
    return False

def simple_explore(prev_collision: bool, t: int) -> str:
    if prev_collision:
        return random.choice(["turn_left", "turn_right"])
    if t % 20 == 0 and random.random() < 0.6:
        return random.choice(["turn_left", "turn_right"])
    return "move_forward"

def _extract_rgb(obs: dict):
    """
    返回 uint8 RGB 图像 [H,W,3]，取不到则返回 None
    """
    if not isinstance(obs, dict):
        return None
    if "rgb" not in obs:
        return None
    rgb = obs["rgb"]
    # habitat 里通常是 uint8 的 HxWx3
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        # 有些配置可能是 float [0,1] 或 [0,255]
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # 有时可能是 3xHxW，做一次兜底
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb

def main(num_episodes=2, max_steps=500):
    cfg = habitat.get_config(CFG_PATH)
    print("=== Final Config Loaded ===")
    print(OmegaConf.to_yaml(cfg), "\n ......")

    env = habitat.Env(cfg)

    # sanity check
    obs = env.reset()
    tmp = env.step({"action": "move_forward"})
    print("step returns type:", type(tmp), "keys:", list(tmp.keys())[:10])

    if SAVE_GIF:
        os.makedirs(GIF_DIR, exist_ok=True)

    for ep in range(num_episodes):
        obs = env.reset()
        episode = env.current_episode

        print(f"\n=== Episode {ep} ===")
        print("scene_id:", episode.scene_id)
        print("start_position:", episode.start_position)

        if hasattr(episode, "goals") and len(episode.goals) > 0:
            print("goal_position:", episode.goals[0].position)

        print("obs keys:", list(obs.keys()))

        prev_collision = False
        traj = []

        # ====== 新增：帧缓存 ======
        frames = []
        if SAVE_GIF:
            first = _extract_rgb(obs)
            if first is not None:
                frames.append(first)
            else:
                print("[WARN] obs 里没有 'rgb'，无法保存 GIF。请检查配置是否开启 RGB sensor。")
        # =========================

        for t in range(max_steps):
            st = env.sim.get_agent_state()
            pos = st.position
            traj.append(pos.copy())

            # pointgoal
            pg = None
            for k in ["pointgoal_with_gps_compass", "pointgoal", "pointgoal_sensor"]:
                if k in obs:
                    pg = obs[k]
                    break

            act = simple_explore(prev_collision, t)

            obs = env.step({"action": act})

            done = env.episode_over
            metrics = env.get_metrics()
            info = metrics

            prev_collision = get_collision(info)
            reward = 0.0

            # ====== 新增：采样存帧 ======
            if SAVE_GIF and (t % RECORD_EVERY == 0) and len(frames) < MAX_FRAMES:
                rgb = _extract_rgb(obs)
                if rgb is not None:
                    frames.append(rgb)
            # ===========================

            if t % 25 == 0 or done:
                print(
                    f"[t={t:03d}] act={act:12s} coll={prev_collision} "
                    f"reward={reward:.3f} done={done} "
                    f"dist={metrics.get('distance_to_goal', None)} "
                    f"spl={metrics.get('spl', None)} success={metrics.get('success', None)} "
                    f"pg_shape={None if pg is None else np.array(pg).shape}"
                )

            if done:
                break

        traj = np.asarray(traj)
        path_len = float(np.linalg.norm(traj[1:] - traj[:-1], axis=1).sum()) if len(traj) > 1 else 0.0
        print(f"approx_path_len: {path_len:.2f} m")

        # ====== 新增：写 GIF ======
        if SAVE_GIF and len(frames) > 1:
            gif_path = os.path.join(GIF_DIR, f"ep{ep:03d}.gif")
            duration = 1.0 / max(FPS, 1)
            imageio.mimsave(gif_path, frames, duration=duration, loop=0)
            print(f"[GIF] saved to: {gif_path}  (frames={len(frames)}, fps={FPS}, every={RECORD_EVERY})")
        # ==========================

    env.close()

if __name__ == "__main__":
    main()
