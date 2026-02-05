import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

import habitat, inspect
from omegaconf import OmegaConf


CFG_PATH = "/data1/user2025263063/project/MARLObjectNav/configs/studyCode/pointnav.yaml"

config_file_Save = "/data1/user2025263063/project/MARLObjectNav/configs/base_Config/01Config.yaml"

outputDir = "/data1/user2025263063/project/MARLObjectNav/result/01"


# 参数
NUM_STEPS = 500
FPS = 20
SAVE_DEPTH= True


def save_rgb(rgb: np.ndarray, path: str):
    """
    Docstring for save_rgb
    
    :param rgb: Description
    :type rgb: np.ndarray (H,W,3)
    :param path: Description
    :type path: str
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def save_depth_vis(depth: np.ndarray, path: str):
    """
    depth: (H,W) or (H,W,1), float32 in meters (often normalized if normalize_depth=true)
    保存一个可视化灰度图（自动拉伸到 0-255）
    """
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    d = depth.astype(np.float32)
    # 忽略无穷/异常值
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    # min-max 拉伸
    dmin, dmax = float(d.min()), float(d.max())
    if dmax - dmin < 1e-6:
        vis = np.zeros_like(d, dtype=np.uint8)
    else:
        vis = ((d - dmin) / (dmax - dmin) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(vis).save(path)



if __name__ == "__main__":
    cfg = habitat.get_config(CFG_PATH)
    print(habitat.get_config)
    print(inspect.getsourcefile(habitat.get_config))
    try:
        with open(config_file_Save,"w",encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
    except OSError as e:
        print(f"写入失败: {e}")
    else:
        print("写入成功", config_file_Save)
    
    print("================基础环境配置================")
    print("CFG_PATH = ", CFG_PATH)
    print("dataset.data_path =", cfg.habitat.dataset.data_path)
    print("simulator.scene =", cfg.habitat.simulator.scene)
    print("simulator.scene_dataset =", cfg.habitat.simulator.scene_dataset)

    # 创建一个环境并加载场景
    print(inspect.getsourcefile(habitat.Env))

    env = habitat.Env(config=cfg)
    print("\nEnvironment created.")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    obs = env.reset()
    print("\nAfter reset:")
    print("Observation keys:", obs.keys())

    frames = []

    rgb0 = obs["rgb"]
    save_rgb(rgb0, os.path.join(outputDir+"/rgb", "frame_0000.png"))
    frames.append(np.array(rgb0, copy=False))

    if SAVE_DEPTH and "depth" in obs:
        save_depth_vis(obs["depth"], os.path.join(outputDir+"/depth", "depth_0000.png"))

    for t in range(1, NUM_STEPS + 1):
        if env.episode_over:
            obs = env.reset()
        action = env.action_space.sample()

        obs = env.step(action)

        rgb = obs["rgb"]
        rgb_path = os.path.join(outputDir+"/rgb", f"frame_{t:04d}.png")
        save_rgb(rgb, rgb_path)
        frames.append(np.array(rgb, copy=False))

        if SAVE_DEPTH and "depth" in obs:
            depth_path = os.path.join(outputDir+"/depth", f"depth_{t:04d}.png")
            save_depth_vis(obs["depth"], depth_path)

        # 可选：打印一下动作
        # print(f"t={t:03d}, action={action}")

    env.close()

    # 写 gif
    gif_path = os.path.join(outputDir+"/gif", "rollout.gif")
    imageio.mimsave(gif_path, frames, fps=FPS)
    print(f"\nSaved frames to: {outputDir}")
    print(f"Saved gif to: {gif_path}")
