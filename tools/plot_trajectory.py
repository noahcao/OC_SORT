"""
    This script is to draw trajectory prediction as in Fig.6 of the paper
"""

import matplotlib.pyplot as plt
import matplotlib
import sys
import numpy as np 
import os

def plot_traj(traj_file, name):
    trajs = np.loadtxt(traj_file, delimiter=",")
    track_ids = np.unique(trajs[:,1])
    for tid in track_ids:
        traj = trajs[np.where(trajs[:,1]==tid)]
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        frames = traj[:100, 0]
        boxes = traj[:100, 2:6]
        boxes_x = boxes[:,0]
        boxes_y = boxes[:,1]
        plt.plot(boxes_x, boxes_y, "ro")
        box_num = boxes_x.shape[0]
        for bind in range(0, box_num-1):
            frame_l = frames[bind]
            frame_r = frames[bind+1]
            box_l = boxes[bind]
            box_r = boxes[bind+1]
            if frame_r == frame_l + 1:
                l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="red")
                ax.add_line(l)
            else:
                l = matplotlib.lines.Line2D([box_l[0], box_r[0]], [box_l[1], box_r[1]], color="gray")
                ax.add_line(l)
        plt.savefig("traj_plots/{}/{}.png".format(name, int(tid)))


if __name__ == "__main__":
    name = sys.argv[1]
    os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)

    gt_src = "datasets/dancetrack/val"

    ours = "path/to/pred/output" # preds
    baseline = "path/to/baseline/output" # baseline outputs
    seqs = os.listdir(gt_src)
    for seq in seqs:
        name = "gt_{}".format(seq)
        os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
        plot_traj(os.path.join(gt_src, seq, "gt/gt.txt"), name)

        name = "baseline_{}".format(seq)
        os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
        plot_traj(os.path.join(baseline, "{}.txt".format(seq)), "baseline_{}".format(seq))

        name = "ours_{}".format(seq)
        os.makedirs(os.path.join("traj_plots/{}".format(name)), exist_ok=True)
        plot_traj(os.path.join(ours, "{}.txt".format(seq)), "ours_{}".format(seq))