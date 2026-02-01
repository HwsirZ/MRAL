import numpy as np

class OccupancyMapper:
    def __init__(self, map_size=20.0, resolution=0.1):
        self.res = resolution
        self.size = int(map_size / resolution)
        self.center = self.size // 2

        self.occ = np.zeros((self.size, self.size), np.uint8)
        self.explored = np.zeros_like(self.occ)

    def update(self, depth, pose, fov=90):
        x, z, yaw = pose
        h, w = depth.shape

        angles = np.linspace(-fov/2, fov/2, w) * np.pi / 180
        for i in range(w):
            d = depth[h//2, i]
            if d <= 0.1:
                continue
            theta = yaw + angles[i]
            wx = x + d * np.cos(theta)
            wz = z + d * np.sin(theta)
            mx = int(wx / self.res) + self.center
            mz = int(wz / self.res) + self.center
            if 0 <= mx < self.size and 0 <= mz < self.size:
                self.occ[mz, mx] = 1
                self.explored[mz, mx] = 1

    def explored_ratio(self):
        return self.explored.sum() / self.explored.size
