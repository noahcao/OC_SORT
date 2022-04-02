"""
    script to convert kitti-format output to bdd-format
"""

import json 
import os 
import sys 
import shutil

# def sanity_check(src, dst):
#     for seq in os.listdir(src):
#         src_file = os.path.join(src, seq)
#         dst_file = os.path.join(dst, seq)
#         src_annos = json.load(open(src_file))
#         dst_annos = json.load(open(dst_file))
#         if not len(src_annos) == len(dst_annos):
#             shutil.copyfile(dst_file, src_file)
#             print(seq)

# if __name__ == "__main__":
#     src, dst = sys.argv[1], sys.argv[2]
#     sanity_check(src, dst)


if __name__ == "__main__":
    src_path, out_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_path, exist_ok=True)
    anno_files = os.listdir(src_path)
    out_dict = {}
    out_dict["config"] = None
    # frames_dict = []
    for anno_f in anno_files:
        video_dict = dict()
        videoName = anno_f.split(".")[0]
        print("convert anno: {}".format(anno_f))
        f = open(os.path.join(src_path, anno_f))
        lines = f.readlines()
        frame_count = 0
        for line in lines:
            terms = line.split()
            frame_id = int(terms[0])
            if frame_count > frame_count + 1:
                # missing frame
                for i in range(frame_count+1, frame_id):
                    frame_name = "%s-%07d.jpg" % (videoName, int(i)+1)
                    frame_entry = {"name": frame_name, "videoName": videoName, 
                         "frameIndex": i}
                    video_dict[i] = dict()
                    video_dict[i]["labels"] = []
                    video_dict[i]["info"] = frame_entry
            frame_count = max(frame_count, frame_id)
            if frame_id not in video_dict:
                video_dict[frame_id] = dict()
            track_id = terms[1]
            cate = terms[2]
            box = terms[6:10]
            x1, y1, x2, y2= [float(d) for d in box]
            score = terms[-1]
            label_entry = {"id": track_id, "score": score, "category": cate,
                "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            frame_name = "%s-%07d.jpg" % (videoName, int(frame_id)+1)
            frame_entry = {"name": frame_name, "videoName": videoName, 
                "frameIndex": frame_id}
            if "info" not in video_dict[frame_id]:
                video_dict[frame_id]["info"] = frame_entry
                video_dict[frame_id]["labels"] = []
            video_dict[frame_id]["labels"].append(label_entry)
        video_labels = []
        frame_ids = video_dict.keys()
        for frame_id in sorted(frame_ids):
            frame_entry = video_dict[frame_id]["info"]
            label_entry = video_dict[frame_id]["labels"]
            video_labels.append({    "name": frame_entry["name"],
                                    "videoName": frame_entry["videoName"],
                                    "frameIndex": frame_entry["frameIndex"],
                                    "labels": label_entry})
        save_path = os.path.join(out_path, "{}.json".format(videoName))
        json.dump(video_labels, open(save_path, 'w'))
    # out_dict["frames"] = frames_dict
    # json.dump(out_dict, open(out_path, "w"))
            
            