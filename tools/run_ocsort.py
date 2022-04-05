from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator, MOTEvaluatorPublic
from utils.args import make_parser

import os
import random
import warnings
import glob
import motmetrics as mmp
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:       
            print(k)     
            logger.info('Comparing {}...'.format(k))
            os.makedirs("results_log", exist_ok=True)
            vflag = open("results_log/eval_{}.txt".format(k), 'w')
            accs.append(mmp.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5, vflag=vflag))
            names.append(k)
            vflag.close()
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1
    cudnn.benchmark = True

    rank = args.local_rank
    """
        This is for MOT17/MOT20 data configuration
    """
    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
        seqs = "MOT17-val"
    elif exp.val_ann == "train_half.json":
        gt_type = '_train_half'
        seqs = "MOT17-train_half"
    elif exp.val_ann == "test.json": 
        gt_type = ''
        seqs = "MOT20-test" if args.mot20 else "MOT17-test"
    else:
        assert 0

    result_folder = "{}_test_results".format(args.expn) if args.test else "{}_results".format(args.expn)
    file_name = os.path.join(exp.output_dir, seqs, result_folder)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)

    if not args.public:
        evaluator = MOTEvaluator(
            args=args,
            dataloader=val_loader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            )
    else:
        evaluator = MOTEvaluatorPublic(
            args=args,
            dataloader=val_loader,
            img_size=exp.test_size,
            confthre=exp.test_conf,
            nmsthre=exp.nmsthre,
            num_classes=exp.num_classes,
            )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    results_folder = os.path.join(file_name, "data")
    os.makedirs(results_folder, exist_ok=True)

    # start evaluate
 
    *_, summary = evaluator.evaluate_ocsort(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )

    logger.info("\n" + summary)
 
    # evaluate MOTA
    mmp.lap.default_solver = 'lap'
    print('gt_type', gt_type)
    gtfiles = glob.glob(
      os.path.join('datasets/mot/train', '*/gt/gt{}.txt'.format(gt_type)))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mmp.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mmp.lap.default_solver))
    logger.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mmp.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mmp.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles if "detections" not in f])    
    
    mh = mmp.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mmp.io.render_summary(summary, formatters=fmt, namemap=mmp.io.motchallenge_metric_names))

    metrics = mmp.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mmp.io.render_summary(summary, formatters=mh.formatters, namemap=mmp.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    exp.output_dir = args.output_dir

    if not args.expn:
        args.expn = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
