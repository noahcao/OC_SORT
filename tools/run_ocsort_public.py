'''
    This script makes tracking over the results of existing
    tracking algorithms. Namely, we run OC-SORT over theirdetections.
    Output in such a way is not strictly accurate because
    building tracks from existing tracking results causes loss
    of detections (usually initializing tracks requires a few
    continuous observations which are not recorded in the output
    tracking results by other methods). But this quick adaptation
    can provide a rough idea about OC-SORT's performance on
    more datasets. For more strict study, we encourage to implement 
    a specific detector on the target dataset and then run OC-SORT 
    over the raw detection results.
    NOTE: this script is not for the reported tracking with public
    detection on MOT17/MOT20 which requires the detection filtering
    following previous practice. See an example from centertrack for
    example: https://github.com/xingyizhou/CenterTrack/blob/d3d52145b71cb9797da2bfb78f0f1e88b286c871/src/lib/utils/tracker.py#L83
'''

from loguru import logger
import time
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

from utils.args import make_parser
import os
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import numpy as np
from trackers.ocsort_tracker.ocsort import OCSort


"""
    BDD has not been supported yet. 
"""
BDD_test_seqs = ['b1c66a42-6f7d68ca', 'b1c81faa-3df17267', 'b1c81faa-c80764c5', 'b1c9c847-3bda4659', 
    'b1ca2e5d-84cf9134', 'b1cac6a7-04e33135', 'b1cd1e94-549d0bfe', 'b1ceb32e-3f481b43', 
    'b1ceb32e-51852abe', 'b1cebfb7-284f5117', 'b1d0091f-75824d0d', 'b1d0091f-f2c2d2ae', 
    'b1d0a191-03dcecc2', 'b1d0a191-06deb55d', 'b1d0a191-28f0e779', 'b1d0a191-2ed2269e', 
    'b1d0a191-5490450b', 'b1d0a191-65deaeef', 'b1d0a191-de8948f6', 'b1d10d08-5b108225', 
    'b1d10d08-743fd86c', 'b1d10d08-c35503b8', 'b1d10d08-da110fcb', 'b1d10d08-ec660956', 
    'b1d22449-117aa773', 'b1d22449-15fb948f', 'b1d22ed6-f1cac061', 'b1d3907b-2278601b', 
    'b1d4b62c-60aab822', 'b1d59b1f-a38aec79', 'b1d7b3ac-0bdb47dc', 'b1d7b3ac-36f2d3b7', 
    'b1d7b3ac-5744370e', 'b1d7b3ac-995f9d8a', 'b1d7b3ac-9e14f05f', 'b1d7b3ac-afa57f22', 
    'b1d968b9-563405f4', 'b1d968b9-ce42734f', 'b1d971b4-ac67ca0d', 'b1d9e136-6c94ea3f', 
    'b1d9e136-9ab25cb3', 'b1dac7f7-6b2e0382', 'b1db7e22-cfa74dc3', 'b1dce572-c6a8cb5e', 
    'b1dd58c1-8b546ba7', 'b1df722f-57d21f3f', 'b1df722f-5bcc3db7', 'b1e0c01d-dd9e6e2f', 
    'b1e1a7b8-0aec80e8', 'b1e1a7b8-65ec7612', 'b1e1a7b8-a7426a97', 'b1e1a7b8-b397c445', 
    'b1e2346e-c5f98707', 'b1e3e9f5-92377424', 'b1e62c91-eca210a9', 'b1e6efc0-2552cc5d', 
    'b1e88fd2-c1e4fd2b', 'b1e8ad72-c3c79240', 'b1e9ee0e-67e26f2e', 'b1ea0ae4-4f770228', 
    'b1eb9133-5cc75c18', 'b1ebfc3c-740ec84a', 'b1ebfc3c-cc9c2bb8', 'b1ee702d-0ae1fc10', 
    'b1ee702d-4a193906', 'b1f022d3-45162c67', 'b1f0efd9-37a14dda', 'b1f0efd9-e900c6e5', 
    'b1f20aa0-3401c3bf', 'b1f20aa0-50213047', 'b1f20aa0-6ef1db42', 'b1f25ff6-1ddb7e43', 
    'b1f4491b-07b32e8c', 'b1f4491b-09593e90', 'b1f4491b-16256d7c', 'b1f4491b-33824f31', 
    'b1f4491b-846d8cb2', 'b1f4491b-97465266', 'b1f4491b-9958bd99', 'b1f4491b-bf7d513f', 
    'b1f4491b-cf446195', 'b1f4491b-d8d1459c', 'b1f4491b-dd8dfed5', 'b1f6c103-5ce1f3c6', 
    'b1f6c103-8b75ea3e', 'b1f6c103-b00e8aad', 'b1f85377-44885085', 'b1fbaab8-68db7df7', 
    'b1fbf878-b31a8293', 'b1fc95c9-644e3c3f', 'b1fc95c9-cb2882c7', 'b1ff4656-0435391e', 
    'b1ff4656-94ee8536', 'b1ff4656-ebcfeb35', 'b200b84e-4a792877', 'b200e97a-bf074435', 
    'b20234fd-822029be', 'b202cae2-672e61c5', 'b202cae2-f46c74a6', 'b2036451-aa924fd1', 
    'b204a5c1-05981158', 'b204a5c1-064b0040', 'b204a5c1-fa3d5b88', 'b205eb4d-f84aaa1a', 
    'b2064e61-2beadd45', 'b206a78b-99f405ab', 'b2080dc7-f9b98a5f', 'b20841f9-cef732d5', 
    'b20b69d2-64b9cdb8', 'b20b69d2-650e674d', 'b20b69d2-6e2b9e73', 'b20b69d2-7767e6b6', 
    'b20b69d2-bd242bf0', 'b20b69d2-ca16c907', 'b20b69d2-e31380a7', 'b20b69d2-ffc1d6af', 
    'b20b9c19-91e01a50', 'b20d494a-cdebe83e', 'b20e291a-32ac11c1', 'b20e291a-6012d836', 
    'b20eae11-149766ce', 'b20eae11-18cd8ca2', 'b20eae11-6817ba7a', 'b20ff95c-b9444127', 
    'b2102d00-5eb86b71', 'b2102d00-a8c09be1', 'b2131b7b-e58faab7', 'b213e4eb-09c01a17', 
    'b214d1e1-f248c616', 'b21547c1-73e457f8', 'b21547c1-796757ac', 'b2156f8e-72e1547c', 
    'b215943a-10e44587', 'b216243d-55963da2', 'b216243d-ad4306b9', 'b2169b74-fa197951', 
    'b21742c2-0e7a2b57', 'b21742c2-18d3463a', 'b2194b15-1825056a', 'b21bfb83-ea32f716', 
    'b21c68e6-65674a17', 'b21c86ac-0dc77d82', 'b21c86ac-2eb7ba16', 'b21c86ac-71205084', 
    'b21d5efb-5e2cd743', 'b2208b0f-2796a692', 'b229488e-e4714bb7', 'b22a4d9f-48b2e986', 
    'b22a4d9f-73cc8810', 'b22e02cd-6af68e18', 'b22f385b-5d7e5202', 'b230132b-ff8f2719', 
    'b230a7b2-c881c382', 'b231a630-c4522992', 'b232c7c9-d251d9ee', 'b2331b83-648e56ca', 
    'b2331b83-a28e6b57', 'b23493b1-3200de1c', 'b237db93-fab44bf2', 'b23a79d1-43dfeecd', 
    'b23a79d1-e434acaa', 'b23adb0d-72704b27', 'b23adb0d-8a7aaced', 'b23b2649-1a78948d', 
    'b23b2649-6af03cd5', 'b23b2649-8349d2a1', 'b23bb45f-ddeea1d8', 'b23c9e00-b425de1b', 
    'b23f7012-32d284ce', 'b23f7012-fab06dac', 'b23fe89b-c704fe97', 'b24071b8-b3ee1196', 
    'b2408e45-984ba5aa', 'b242929f-3051abca', 'b242f6b2-0033bdfb', 'b242f6b2-99d2f2c1', 
    'b242f6b2-eaa39345', 'b242f6b2-f5da110f', 'b24380ab-63272e5a', 'b24380ab-6dbeb908', 
    'b24c9ee6-e43a6e8b', 'b24ca67a-594d7d3c', 'b24d283f-33783d1b', 'b24f03f7-ff66eaca', 
    'b24f7455-e8c55d6a', 'b2505382-1423f56a', 'b2505382-272e7823', 'b2505382-2905b23c', 
    'b2505382-549785d3', 'b2505382-de5238f0', 'b250fb0c-01a1b8d3', 'b251064f-30002542', 
    'b251064f-4696b75e', 'b251064f-5f6b663e', 'b251064f-8d92db81', 'b251064f-e7a165fd', 
    'b251b746-00138418', 'b255cd6c-0bdf0ac7', 'b255cd6c-2f889586', 'b255cd6c-5ccba454']


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(args):
    results_folder = args.out_path
    raw_path = args.raw_results_path
    os.makedirs(results_folder, exist_ok=True)

    dataset = args.dataset

    total_time = 0 
    total_frame = 0 

    if dataset == "kitti":
        test_seqs = ["%04d" % i for i in range(29)]
        cats = ['Pedestrian', 'Car', 'Cyclist', "Van", "Truck"]
    elif dataset == "bdd":
        """
            We are not supporting BDD yet. This is a placeholder for now.
        """
        test_seqs = BDD_test_seqs
        cats = ["rider", "car", "truck", "bicycle", "motorcycle", "pedestrian", "bus"]
    elif dataset == "headtrack":
        test_seqs = ["HT21-11", "HT21-12", "HT21-13", "HT21-14", "HT21-15"]
        cats = ["head"]
    else:
        assert(0)

    cat_ids = {cat: i for i, cat in enumerate(cats)}

    for seq_name in test_seqs:
        print("starting seq {}".format(seq_name))
        tracker = OCSort(args.track_thresh, iou_threshold=args.iou_thresh, delta_t=args.deltat, 
            asso_func=args.asso, inertia=args.inertia)
        if dataset in ["kitti", "bdd"]:
            seq_trks = np.empty((0, 18))
        elif dataset == "headtrack":
            seq_trks = np.empty((0, 10))
        seq_file = os.path.join(raw_path, "{}.txt".format(seq_name))
        seq_file = open(seq_file)
        out_file = os.path.join(results_folder, "{}.txt".format(seq_name))
        out_file = open(out_file, 'w')
        lines = seq_file.readlines()
        line_count = 0 
        for line in lines:
            print("{}/{}".format(line_count,len(lines)))
            line_count+=1
            line = line.strip()
            if dataset in ["kitti", "bdd"]:
                tmps = line.strip().split()
                tmps[2] = cat_ids[tmps[2]]
            elif dataset == "headtrack":
                tmps = line.strip().split(",")
            trk = np.array([float(d) for d in tmps])
            trk = np.expand_dims(trk, axis=0)
            seq_trks = np.concatenate([seq_trks, trk], axis=0)
        min_frame = seq_trks[:,0].min()
        max_frame = seq_trks[:,0].max()
        for frame_ind in range(int(min_frame), int(max_frame)+1):
            print("{}:{}/{}".format(seq_name, frame_ind, max_frame))
            if dataset in ["kitti", "bdd"]:
                dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6:10]
                cates = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2]
                scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,-1]
            elif dataset == "headtrack":
                dets = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,2:6]
                cates = np.zeros((dets.shape[0],))
                scores = seq_trks[np.where(seq_trks[:,0]==frame_ind)][:,6]
                dets[:, 2:] += dets[:, :2] # xywh -> xyxy
            else:
                assert(0)
            assert(dets.shape[0] == cates.shape[0])
            t0 = time.time()
            online_targets = tracker.update_public(dets, cates, scores)
            t1 = time.time()
            total_frame += 1
            total_time += t1 - t0
            trk_num = online_targets.shape[0]
            boxes = online_targets[:, :4]
            ids = online_targets[:, 4]
            frame_counts = online_targets[:, 6]
            sorted_frame_counts = np.argsort(frame_counts)
            frame_counts = frame_counts[sorted_frame_counts]
            cates = online_targets[:, 5]
            cates = cates[sorted_frame_counts].tolist()
            cates = [cats[int(catid)] for catid in cates]
            boxes = boxes[sorted_frame_counts]
            ids = ids[sorted_frame_counts]
            for trk in range(trk_num):
                lag_frame = frame_counts[trk]
                if frame_ind < 2*args.min_hits and lag_frame < 0:
                    continue
                """
                    NOTE: here we use the Head Padding (HP) strategy by default, disable the following
                    lines to revert back to the default version of OC-SORT.
                """
                if dataset in ["kitti", "bdd"]:
                    out_line = "{} {} {} -1 -1 -1 {} {} {} {} -1 -1 -1 -1000 -1000 -1000 -10 1\n".format\
                        (int(frame_ind+lag_frame), int(ids[trk]), cates[trk], 
                        boxes[trk][0], boxes[trk][1], boxes[trk][2], boxes[trk][3])
                elif dataset == "headtrack":
                    out_line = "{},{},{},{},{},{},{},-1,-1,-1\n".format(int(frame_ind+lag_frame), int(ids[trk]),
                        boxes[trk][0], boxes[trk][1], 
                        boxes[trk][2]-boxes[trk][0],
                        boxes[trk][3]-boxes[trk][1], 1)
                out_file.write(out_line)

    print("Running over {} frames takes {}s. FPS={}".format(total_frame, total_time, total_frame / total_time))
    return 


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)