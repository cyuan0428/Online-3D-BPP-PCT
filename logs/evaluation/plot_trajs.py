import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# 請根據你的實際路徑修改這行
trajs_path = "logs/evaluation/test1-2025.07.29-13-12-40/trajs.npy"
outdir = "logs/evaluation/test1-2025.07.29-13-12-40/packing_figs"
os.makedirs(outdir, exist_ok=True)

data = np.load(trajs_path, allow_pickle=True)
print(f"共 {len(data)} 集，將逐一輸出圖檔到 {outdir}/")

def draw_box(ax, x, y, z, w, h, d, color='skyblue'):
    corners = [
        [x, y, z], [x+w, y, z], [x+w, y+h, z], [x, y+h, z],
        [x, y, z+d], [x+w, y, z+d], [x+w, y+h, z+d], [x, y+h, z+d]
    ]
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[4,7,3,0]]
    for face in faces:
        ax.add_collection3d(Poly3DCollection([ [corners[i] for i in face] ], facecolors=color, edgecolors='k', alpha=0.7))

for idx, episode in enumerate(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for box in episode:
        # 若你的 box 不是 dict 而是 array，請用 box[0], box[1], ... 替換
        x = box['x'] if isinstance(box, dict) else box[0]
        y = box['y'] if isinstance(box, dict) else box[1]
        z = box['z'] if isinstance(box, dict) else box[2]
        w = box['w'] if isinstance(box, dict) else box[3]
        h = box['h'] if isinstance(box, dict) else box[4]
        d = box['d'] if isinstance(box, dict) else box[5]
        draw_box(ax, x, y, z, w, h, d)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"Episode {idx} Packing Result")
    plt.tight_layout()
    outname = os.path.join(outdir, f"episode_{idx:03d}.png")
    plt.savefig(outname)
    plt.close(fig)
    print(f"Episode {idx} → 已存成 {outname}")

print("全部完成！每集一張圖存在 packing_figs/ 資料夾。")
