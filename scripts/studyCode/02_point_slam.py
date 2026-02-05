import os, math
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import quaternion

import habitat, inspect
from omegaconf import OmegaConf


CFG_PATH = "/data1/user2025263063/project/MARLObjectNav/configs/studyCode/point_slam.yaml"

config_file_Save = "/data1/user2025263063/project/MARLObjectNav/configs/base_Config/02Config.yaml"

outputDir = "/data1/user2025263063/project/MARLObjectNav/result/02"


NUM_STEPS = 500 
RES = 0.10      # 每个栅格的大小 
MAX_DEPTH = 10.0
FREE_VAL = 1
OCC_VAL  = 2

# Policy
GOAL_REFRESH_EVERY  = 80 
GOAL_REACHED_THRESH = 0.5
MOVE_FORWARD_PROB   = 0.7   # 选择前进的概率


# scripts 保存/拼图
def to_uint8_rgb(rgb):
    """
    强制转换 rgb 图像数据类型为 uint8
    :param rgb: rgb 数据 大小 256*256 
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb,0,255).astype(np.uint8)
    return rgb

def depth_to_vis(depth):
    """
    将深度图转换为能直接保存显式的 8_bit 灰度图格式
    
    :param depth: 大小 (256, 256, 1) type float32
    """

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]       # 取出唯一通道
    # 处理异常值, NAN +inf -inf 转换为0
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0) 
    # 对深度进行截断, 只看 MAX_DEPTH 范围, 也就是限定视野范围
    d = np.clip(d, 0.0, MAX_DEPTH)  

    if d.max() - d.min() < 1e-6:    # 全零或者值相近 无有效动态范围
        vis = np.zeros_like(d, dtype=np.uint8)   # 返回全黑图 
    else:
        # min-max 标准归一化映射到 8_bit空间
        vis = ((d - d.min()) / (d.max() - d.min()) * 255.0).astype(np.uint8)  
    return vis

def save_image(arr, path):
    """
    将一个 numpy 数组 arr 转为 PIL.image 并保存
    
    :param arr: numpy arr
    :param path: save path
    """
    Image.fromarray(arr).save(path)

def cat_h(img_left, img_right):
    """
    将两张 RGB 图像在水平方向上拼接成一张更宽的图
    
    :param img_left: numpy arr
    :param img_right: numpy arr
    """

    # both uint8 images (H,W,3)
    h = max(img_left.shape[0], img_right.shape[0]) # 取最大高度
    # 将高度对齐
    def pad_to_h(img):
        if img.shape[0] == h:
            return img
        pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])
    L = pad_to_h(img_left)
    R = pad_to_h(img_right)
    return np.hstack([L, R])


# Occupancy grid：建图
def yaw_from_quat_xyzw(q):
    """
    将四元姿态 q=[x,y,z,w] 转换为绕竖直轴的偏航角yaw, 也就是机器人
    在水平面的朝向角
    
    :param q: Description
    """
    """
    导航中的姿态:
        roll:   绕 x 轴 (左右翻滚)
        pitch:  绕 z 或者 x (抬头/低头)
        yaw:    绕 竖直轴 (平面内转向)
    """
    # q: [x,y,z,w]
    # x, y, z, w = (q.x, q.y, q.z, q.w)
    # # 转换公式
    # siny_cosp = 2.0 * (w * y + x * z)
    # cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    # return math.atan2(siny_cosp, cosy_cosp)
    # Habitat forward in local frame is -Z
    f_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    f_world = quaternion.rotate_vectors(q, f_local)  # shape (3,)
    # heading angle in x-z plane: angle from +x toward +z
    return math.atan2(float(f_world[2]), float(f_world[0]))


def init_grid_from_pathfinder(sim, res):
    """
    用 habitat 的 nvmesh 边界, 在 x-z 平面初始化一个 2D 栅格地图
    
    :param sim: Habitat-sim 的 simulator, 有 pathfinder
    :param res: 地图的分辨率 resolution
    """

    # 用 navmesh bounds 做地图范围
    # bounds: (lower, upper), each is (x,y,z)
    lower, upper = sim.pathfinder.get_bounds()  # 返回nvmash 的轴对齐 bound-box
    # 用 x,z 平面做 2D grid
    x_min, x_max = lower[0], upper[0]
    z_min, z_max = lower[2], upper[2]
    # 计算栅格尺寸
    W = int(math.ceil((x_max - x_min) / res)) + 1
    H = int(math.ceil((z_max - z_min) / res)) + 1
    # 初始化为全零
    grid = np.zeros((H, W), dtype=np.uint8)  # 0 unknown, 1 free, 2 occ
    origin = (x_min, z_min)
    return grid, origin

def world_to_grid(x, z, origin, res):
    """
    世界坐标到栅格坐标的转换
    
    :param x: world position
    :param z: world position
    :param origin: 栅格地图原点在世界坐标的位置
    :param res: 分辨率, 单个栅格宽度
    """
    """
    habitat 的世界坐标系是右手系 (right-handed)
    传感器/相机坐标系是 OpenGL 风格 +X 向右 +Y 向上 -z 朝前
    """
    x0, z0 = origin
    gx = int((x - x0) / res)
    gz = int((z - z0) / res)
    return gx, gz


def bresenham(x0, y0, x1, y1):
    """
    bresenham 直线栅格化算法, 在 grid 上从(x0,y0) 到 (x1,y1) 生成一条接近真实直线的离散格点序列
    
    :param x0: 起点
    :param y0: 起点
    :param x1: 终点
    :param y1: 终点
    """
    # integer grid line
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def integrate_depth_to_grid(grid, origin, res, agent_pos, agent_yaw, depth, hfov_deg):
    """
    将深度图粗略融合到 2D 栅格 grid 中
    
    :param grid: grid
    :param origin: 世界坐标中地图原点
    :param res: resolution
    :param agent_pos: world position 
    :param agent_yaw: 朝向
    :param depth: RGBD image
    :param hfov_deg: 水平视场角
    """

    # 预处理 depth, 取维数, 转 float
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    d = depth.astype(np.float32)

    # 若 normalize_depth=true，Habitat depth 往往是 [0,1]，映射到 [0,max_depth]
    # 你 config 里 max_depth=10, normalize_depth=true
    # 归一化深度
    d = np.clip(d, 0.0, 1.0) * 10
    d = np.minimum(d, MAX_DEPTH)

    H, W = d.shape
    hfov = math.radians(hfov_deg)
    fx = (W / 2.0) / math.tan(hfov / 2.0)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # 下采样以提速
    stride = 4
    xs = np.arange(0, W, stride)
    ys = np.arange(0, H, stride)

    ax, ay, az = agent_pos  # world
    gx0, gz0 = world_to_grid(ax, az, origin, res)

    cos_y = math.cos(agent_yaw)
    sin_y = math.sin(agent_yaw)

    for v in ys:
        for u in xs:
            z_cam = d[v, u]
            if z_cam <= 1e-3:
                continue

            # 相机坐标系：x右 y下 z前（这里用常见 pinhole 近似）
            x_cam = (u - cx) * z_cam / fx
            # y_cam = (v - cy) * z_cam / fx  # 我们只用地面投影，忽略高度

            # 只做平面投影：把相机前向 z_cam 与横向 x_cam 旋转到世界 xz
            # 相机与机体朝向一致的近似：world_dx = cos*y * z - sin*y * x, world_dz = sin*y * z + cos*y * x
            dx = cos_y * z_cam - sin_y * x_cam
            dz = sin_y * z_cam + cos_y * x_cam

            x_w = ax + dx
            z_w = az + dz

            gx1, gz1 = world_to_grid(x_w, z_w, origin, res)

            # 边界检查
            if not (0 <= gz0 < grid.shape[0] and 0 <= gx0 < grid.shape[1]):
                continue
            if not (0 <= gz1 < grid.shape[0] and 0 <= gx1 < grid.shape[1]):
                continue

            pts = bresenham(gx0, gz0, gx1, gz1)
            # 射线除最后一个点标 free
            for (px, pz) in pts[:-1]:
                if 0 <= pz < grid.shape[0] and 0 <= px < grid.shape[1]:
                    if grid[pz, px] == 0:
                        grid[pz, px] = FREE_VAL
            # 终点标 occupied
            px, pz = pts[-1]
            if 0 <= pz < grid.shape[0] and 0 <= px < grid.shape[1]:
                grid[pz, px] = OCC_VAL


def render_grid(grid, traj_cells=None):
    # 0 unknown -> 黑, 1 free -> 灰, 2 occ -> 白, traj -> 红
    H, W = grid.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[grid == 1] = (80, 80, 80)
    img[grid == 2] = (230, 230, 230)
    if traj_cells is not None:
        for (gx, gz) in traj_cells[-2000:]:
            if 0 <= gz < H and 0 <= gx < W:
                img[gz, gx] = (255, 0, 0)
    # 让图像更大便于看
    scale = 2
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return img 


# 简单 "规划式探索"：随机采样目标 + 最短路跟踪
def sample_navigable_goal(sim):
    p = sim.pathfinder.get_random_navigable_point()
    return np.array([p[0], p[1], p[2]], dtype=np.float32)

def plan_shortest_path(sim, start, goal):
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = start
    sp.requested_end = goal
    found = sim.pathfinder.find_path(sp)
    if not found or len(sp.points) == 0:
        return None
    return np.array(sp.points, dtype=np.float32)

def angle_wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def choose_action_to_waypoint(agent_pos, agent_yaw, wp, turn_thresh=0.2):
    # 计算当前朝向与目标方向夹角，决定 turn 或 forward
    dx = wp[0] - agent_pos[0]
    dz = wp[2] - agent_pos[2]
    desired = math.atan2(dz, dx)
    diff = angle_wrap(desired - agent_yaw)
    if diff > turn_thresh:
        return {"action": "turn_right"}
    if diff < -turn_thresh:
        return {"action": "turn_left"}
    return {"action": "move_forward"}

def topdown_map_to_rgb(td):
    """
    td: (H,W) uint8, 值域通常 0..4
    """
    td = td.astype(np.uint8)
    H, W = td.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # 经验映射：不同版本含义可能略差，但视觉会很清晰
    out[td == 0] = (0, 0, 0)         # unknown/背景
    out[td == 1] = (60, 60, 60)      # free / navigable
    out[td == 2] = (220, 220, 220)   # occupied / obstacle / border
    out[td == 3] = (0, 255, 0)       # source / start（如果有）
    out[td == 4] = (255, 0, 0)       # goal / path（如果有）

    return out




def main():
    cfg = habitat.get_config(CFG_PATH)

    env = habitat.Env(config=cfg)

    try:
        with open(config_file_Save,"w",encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
    except OSError as e:
        print(f"写入失败: {e}")
    else:
        print("写入成功", config_file_Save)
    sim = env._sim                  # HabitatSim

    obs = env.reset()

    # 初始化 grid
    grid, origin = init_grid_from_pathfinder(sim, RES)
    traj = []

    # 目标与路径
    goal = None
    path = None
    path_i = 0

    frames = []

    # 从 sensor config 读 hfov（你的是 90）
    hfov = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov

    for t in range(NUM_STEPS):
        if env.episode_over:
            print("restart")
            obs = env.reset()
            goal, path, path_i = None, None, 0

        # 当前位姿
        st = sim.get_agent_state()
        # print("st type:", type(st))
        # print("st:",st)
        pos = np.array(st.position, dtype=np.float32)
        # print("st.rotation:",st.rotation)
        # print("st.rotation type:",type(st.rotation))
        yaw = yaw_from_quat_xyzw(st.rotation)

        # 更新轨迹到 grid
        gx, gz = world_to_grid(pos[0], pos[2], origin, RES)
        traj.append((gx, gz))

        # 建图（depth）
        integrate_depth_to_grid(
            grid, origin, RES,
            agent_pos=pos, agent_yaw=yaw,
            depth=obs["depth"], hfov_deg=hfov
        )

        # 规划式探索：每隔 K 步刷新一个随机目标 + 最短路径
        if (goal is None) or (t % GOAL_REFRESH_EVERY == 0) or (path is None):
            goal = sample_navigable_goal(sim)
            path = plan_shortest_path(sim, pos, goal)
            path_i = 0

        # 若规划失败，就随机走
        if path is None or len(path) < 2:
            act = {"action": np.random.choice(["move_forward", "turn_left", "turn_right"], p=[MOVE_FORWARD_PROB, (1-MOVE_FORWARD_PROB)/2, (1-MOVE_FORWARD_PROB)/2])}
        else:
            # 走向下一个 waypoint
            # 跳过已经接近的点
            while path_i < len(path) and np.linalg.norm(path[path_i][[0,2]] - pos[[0,2]]) < 0.3:
                path_i += 1
            if path_i >= len(path):
                act = {"action": "turn_left"}
            else:
                act = choose_action_to_waypoint(pos, yaw, path[path_i])

        # step
        print("steps:",t)
        print("action->",act)

        # 在 step 前
        # print("yaw=", yaw, "pos=", pos[[0,2]], "wp=", path[path_i][[0,2]])
        # dx = path[path_i][0] - pos[0]
        # dz = path[path_i][2] - pos[2]
        # desired = math.atan2(dz, dx)
        # diff = angle_wrap(desired - yaw)
        # print("desired=", desired, "diff=", diff)

        obs = env.step(act)

        # if t < 3:
        #     m = env.get_metrics()
        #     tdm = m.get("top_down_map", None)
        #     print("has top_down_map:", tdm is not None, "type:", type(tdm))

        #     if isinstance(tdm, dict):
        #         print("tdm keys:", list(tdm.keys()))
        #         arr = tdm.get("map", None)
        #         print("map exists:", arr is not None)
        #         if arr is not None:
        #             arr = np.array(arr)
        #             print("map shape:", arr.shape, "dtype:", arr.dtype,
        #                 "min/max:", float(arr.min()), float(arr.max()))
        #         # 额外：看看 fog mask
        #         if "fog_of_war_mask" in tdm:
        #             f = np.array(tdm["fog_of_war_mask"])
        #             print("fog mask:", f.shape, f.dtype, float(f.min()), float(f.max()))
        #     else:
        #         arr = np.array(tdm)
        #         print("tdm array shape:", arr.shape, "dtype:", arr.dtype,
        #             "min/max:", float(arr.min()), float(arr.max()))


        


        # # 可视化帧：左 rgb，右 grid
        # rgb = to_uint8_rgb(obs["rgb"])
        # grid_img = render_grid(grid, traj_cells=traj)
        # # 把 grid 变到 3通道并 resize 到和 rgb 差不多高度
        # grid_pil = Image.fromarray(grid_img).resize((rgb.shape[0], rgb.shape[0]))
        # grid_vis = np.array(grid_pil.convert("RGB"))

        # canvas = cat_h(rgb, grid_vis)
        # frames.append(canvas)

        # save_image(canvas, os.path.join(outputDir+"/rgb", f"frame_{t:04d}.png"))

        rgb = to_uint8_rgb(obs["rgb"])

        # grid 可视化
        grid_img = render_grid(grid, traj_cells=traj)
        grid_pil = Image.fromarray(grid_img).resize((rgb.shape[1], rgb.shape[0]))
        grid_vis = np.array(grid_pil.convert("RGB"))

        # top-down 可视化
        metrics = env.get_metrics()
        tdm = metrics.get("top_down_map", None)

        if isinstance(tdm, dict) and tdm.get("map", None) is not None:
            td = np.array(tdm["map"], dtype=np.uint8)  # (H,W), 值域 0..4
            td_rgb = topdown_map_to_rgb(td)
            # 拉伸到 0..255 并转 3 通道
            # td_vis = np.repeat((td * 63).astype(np.uint8)[..., None], 3, axis=-1)
            td_vis = np.array(Image.fromarray(td_rgb).resize((rgb.shape[1], rgb.shape[0]),resample=Image.NEAREST))
        else:
            td_vis = np.zeros_like(rgb)


        canvas = np.hstack([rgb, grid_vis, td_vis])
        frames.append(canvas)
        save_image(canvas, os.path.join(outputDir, "rgb", f"frame_{t:04d}.png"))



    env.close()

    gif_path = os.path.join(outputDir+"/gif", "explore.gif")
    imageio.mimsave(gif_path, frames, fps=20)
    print("Saved:", gif_path)

if __name__ == "__main__":
    main()


