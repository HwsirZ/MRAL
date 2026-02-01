import cv2
import numpy as np

class Recorder:
    def __init__(self, path="out.mp4"):
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 512)
        )

    def add(self, rgb, depth, occ):
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        occ_vis = (occ * 255).astype(np.uint8)
        occ_vis = cv2.cvtColor(occ_vis, cv2.COLOR_GRAY2BGR)

        frame = np.concatenate([
            np.concatenate([rgb, occ_vis], axis=1),
            np.concatenate([cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR), occ_vis], axis=1)
        ], axis=0)

        self.writer.write(frame)

    def close(self):
        self.writer.release()
