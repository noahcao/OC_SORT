"""
    script to convert prediction files in BDD100k format into KITTI format,
    considering to attend the BDD100k challenge for more information:
    https://www.bdd100k.com. We haven't run OC-SORT on BDD100K yet. Will likely
    to update that later.
"""
import json 
import sys 
import os


# Example: 0 1 Car -1 -1 -1 483.81 173.31 658.93 242.23 -1 -1 -1 -1000 -1000 -1000 -10 0.93
KITTI_format = "%d %d %s -1 -1 -1 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n"

seen_cate = []

def write_preds(summary, out_path):
    os.makedirs(out_path, exist_ok=True)
    video_names = summary.keys()
    for video in video_names:
        f = open(os.path.join(out_path, "{}.txt".format(video)), 'w')
        labels = summary[video]
        frames = labels.keys()
        max_frame = max(frames)
        min_frame = min(frames)
        for f_idx in range(min_frame, max_frame+1):
            # print("writing seq: {}: {}/{}".format(video, f_idx, max_frame))
            frame_label = labels[f_idx]
            if frame_label is None:
                continue
            else:
                for entry in frame_label:
                    track_id = int(entry["id"])
                    score = float(entry["score"])
                    cate = entry["category"]
                    box = entry["box2d"]
                    x1, x2, y1, y2 = box["x1"], box["x2"], box["y1"], box["y2"]
                    x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)
                    write_line = KITTI_format % (f_idx, track_id, \
                            cate, x1, y1, x2, y2, score)
                    f.write(write_line)



def convert_to_kitti(annos):
    video_dict = dict()
    for ann in annos:
        videoName = ann["videoName"]
        frameIndex = ann["frameIndex"]
        if videoName not in video_dict:
            video_dict[videoName] = dict()
        if "labels" in ann.keys():
            labels = ann["labels"]
        else:
            labels = None
        video_dict[videoName][frameIndex] = labels 
    return video_dict


# if __name__ == "__main__":
#     # for qdtrack results
#     src_file, out_path = sys.argv[1], sys.argv[2]
#     preds = json.load(open(src_file))["frames"]
#     summary = convert_to_kitti(preds)
#     write_preds(summary, out_path)

if __name__ == "__main__":
    src_path, out_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_path, exist_ok=True)
    results = os.listdir(src_path)
    for result in results:
        video_annos = []
        seq_name = result.split(".")[0]
        save_path = os.path.join(out_path, "{}.txt".format(seq_name))
        f = open(save_path, "w")
        src_annos = json.load(open(os.path.join(src_path, result)))
        out_annos = convert_to_kitti(src_annos)[seq_name]
        frames = sorted(out_annos.keys())
        for frame in frames:
            tracks = out_annos[frame]
            for entry in tracks:
                track_id = int(entry["id"])
                # score = float(entry["score"])
                cate = entry["category"]
                box = entry["box2d"]
                x1, x2, y1, y2 = box["x1"], box["x2"], box["y1"], box["y2"]
                x1, x2, y1, y2 = float(x1), float(x2), float(y1), float(y2)
                write_line = KITTI_format % (frame, track_id, \
                        cate, x1, y1, x2, y2, 1)
                f.write(write_line)
