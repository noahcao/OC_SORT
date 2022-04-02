# Get Started 
We introduce the process of getting started on OC-SORT. This instruction is adapted from ByteTrack especially for the training part. We provide some simple pieces here, for details please refer to the source code and *utils/args.py*.

## Data preparation

1. Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [DanceTrack](https://github.com/DanceTrack/DanceTrack) and put them under <OCSORT_HOME>/datasets in the following structure:
    ```
    datasets
    |——————mot
    |        └——————train
    |        └——————test
    └——————crowdhuman
    |        └——————Crowdhuman_train
    |        └——————Crowdhuman_val
    |        └——————annotation_train.odgt
    |        └——————annotation_val.odgt
    └——————MOT20
    |        └——————train
    |        └——————test
    └——————Cityscapes
    |        └——————images
    |        └——————labels_with_ids
    └——————ETHZ
    |        └——————eth01
    |        └——————...
    |        └——————eth07
    └——————dancetrack        
             └——————train
             └——————val
             └——————test
    ```

2. Turn the datasets to COCO format and mix different training data:

    ```python
    # replace "dance" with ethz/mot17/mot20/crowdhuman/cityperson for others
    python3 tools/convert_dance_to_coco.py 
    ```

3. *[Optional]* If you want to training for MOT17/MOT20, follow the following to create mixed training set.

    ```python
    # build mixed training sets for MOT17 and MOT20 
    python3 tools/mix_data_{ablation/mot17/mot20}.py
    ```

## Training
You can use OC-SORT without training by adopting existing detectors. But we borrow the training guidelines from ByteTrack in case you want work on your own detector. 

Download the COCO-pretrained YOLOX weight [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.1.0) and put it under *\<OCSORT_HOME\>/pretrained*.

* **Train ablation model (MOT17 half train and CrowdHuman)**

    ```shell
    python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
    ```

* **Train MOT17 test model (MOT17 train, CrowdHuman, Cityperson and ETHZ)**

    ```shell
    python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
    ```

* **Train MOT20 test model (MOT20 train, CrowdHuman)**

    For MOT20, you need to uncomment some code lines to add box clipping: [[1]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment),[[2]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122),[[3]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217) and [[4]](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115). Then run the command:

    ```shell
    python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
    ```

* **Train on DanceTrack train set**
    ```shell
    python3 tools/train.py -f exps/example/dancetrack/yolox_x.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
    ```

* **Train custom dataset**

    First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

    ```shell
    python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
    ```


## Evaluation

* **on DanceTrack Val set**
    ```shell
    python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_val.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --expn $exp_name
    ```
    We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the officially released validation set. This gives HOTA = 52.1 ~ 52.6.

* **on DanceTrack Test set**
    ```shell
    python tools/run_ocsort_dance.py -f exps/example/mot/yolox_dancetrack_test.py -c pretrained/bytetrack_dance_model.pth.tar -b 1 -d 1 --fp16 --fuse --test --expn $exp_name
    ```
    Submit the outputs to [the DanceTrack evaluation site](https://competitions.codalab.org/competitions/35786). This gives HOTA = 54.6 ~ 55.2.

* **on MOT17 half val**
    ```shell
    python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse --expn $exp_name
    ```
    We follow the [TrackEval protocol](https://github.com/DanceTrack/DanceTrack/tree/main/TrackEval) for evaluation on the self-splitted validation set. This gives you HOTA = 66.5.

* **on MOT17/MOT20 Test set**
    ```shell
    # MOT17
    python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse --expn $exp_name

    # MOT20
    python3 tools/run_ocsort.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -c pretrained/bytetrack_x_mot20.tar -b 1 -d 1 --fp16 --fuse --track_thresh 0.4 --mot20 --expn $exp_name
    ```
    Submit the zipped output files to [MOTChallenge](https://motchallenge.net/) system. Following [the adaptive detection thresholds](https://github.com/ifzhang/ByteTrack/blob/d742a3321c14a7412f024f2218142c7441c1b699/yolox/evaluators/mot_evaluator.py#L139) by ByteTrack can further boost the performance. After interpolation (see below), this gives you HOTA = ~63.1 on MOT17 and HOTA = \~61.9 on MOT20.

* **on KITTI Test set**
  
    For fair comparison of association performance only, we use the detection results from a public [PermaTrack](https://github.com/TRI-ML/permatrack) model weights as indicated [here](https://github.com/TRI-ML/permatrack/issues/16). The output of PermanceTrack is provided in *exps/permatrack_kitti_test/* already. Simply run the command to use OC-SORT over KITTI-test set:

    ```shell
    python tools/run_ocsort_public.py  --hp --out_path kitti_test --dataset kitti --raw_results_path exps/permatrack_kitti_test
    ```

    Then, submit the output .txt files to [the KITTI evaluation server](http://www.cvlibs.net/datasets/kitti/).

* **on HeadTrack21 (CroHD)**

    Similar to KITTI, we do not provide detector on CroHD head tracking dataset but adapt from detection results from existing results. Please download the results reported to the [evaluation server](https://motchallenge.net/results/Head_Tracking_21/) and place it at *datasets/headtrack/$tracker*

    ```shell
    python tools/run_ocsort_public.py  --raw_results_path datasets/headtrack/$tracker --hp --dataset headtrack --out_path out_headtrack_results 
    ```
    Then, submit the output files to the evaluation server above for evaluation.


*Note: We find the current implementation may show some randomness in different running trials. We are still inspecting this.*

## [Optional] Interpolation
OC-SORT is designed for online tracking, but offline interpolation has been demonstrated efficient for many cases. To use the linear interpolation over existing tracking results:
```shell
    # optional offline post-processing
    python3 tools/interpolation.py $result_path $save_path
```
Furthermore, we provide a piece of attempt of using Gaussian Process Regression in interpolating trajectories, which work upon existing linear interpolation results:
```shell
    python3 tools/gp_interpolation.py $raw_results_path $linear_interp_path $save_path
```
*Note: for the results in our paper on MOT17/MOT20 private settings and HeadTrack, we use linear interpolation by default.*