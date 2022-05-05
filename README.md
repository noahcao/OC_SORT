# OC-SORT

[![arXiv](https://img.shields.io/badge/arXiv-2203.14360-<COLOR>.svg)](https://arxiv.org/abs/2203.14360) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![test](https://img.shields.io/static/v1?label=By&message=Pytorch&color=red)

**Observation-Centric SORT (OC-SORT)** is a pure motion-model-based multi-object tracker. It aims to improve tracking robustness in **crowded scenes and when objects are in non-linear motion**. It is designed by recognizing and fixing limitations in Kalman filter and [SORT](https://arxiv.org/abs/1602.00763). It is flexible to integrate with different detectors and matching modules, such as appearance similarity. It remains, **Simple, Online and Real-time**.


<center>
<img src="assets/teaser.png" width="600"/>
</center>

## News
* [04/27/2022]: Support intergration with BYTE and multiple cost metrics, such as GIoU, CIoU, etc.
* [04/02/2022]: A preview version is released after a primary cleanup and refactor. 
* [03/27/2022]: The [arxiv preprint](https://arxiv.org/abs/2203.14360) of OC-SORT is released.

## Benchmark Performance

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/observation-centric-sort-rethinking-sort-for/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=observation-centric-sort-rethinking-sort-for) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/observation-centric-sort-rethinking-sort-for/multiple-object-tracking-on-kitti-tracking)](https://paperswithcode.com/sota/multiple-object-tracking-on-kitti-tracking?p=observation-centric-sort-rethinking-sort-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/observation-centric-sort-rethinking-sort-for/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=observation-centric-sort-rethinking-sort-for) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/observation-centric-sort-rethinking-sort-for/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=observation-centric-sort-rethinking-sort-for) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/observation-centric-sort-rethinking-sort-for/multiple-object-tracking-on-crohd)](https://paperswithcode.com/sota/multiple-object-tracking-on-crohd?p=observation-centric-sort-rethinking-sort-for)


| Dataset          | HOTA | AssA | IDF1 | MOTA | FP      | FN      | IDs   | Frag   |
| ---------------- | ---- | ---- | ---- | ---- | ------- | ------- | ----- | ------ |
| MOT17 (private)  | 63.2 | 63.2 | 77.5 | 78.0 | 15,129  | 107,055 | 1,950 | 2,040  |
| MOT17 (public)   | 52.4 | 57.6 | 65.1 | 58.2 | 4,379   | 230,449 | 784   | 2,006  |
| MOT20 (private)  | 62.4 | 62.5 | 76.4 | 75.9 | 20,218  | 103,791 | 938   | 1,004  |
| MOT20 (public)   | 54.3 | 59.5 | 67.0 | 59.9 | 4,434   | 202,502 | 554   | 2,345  |
| KITTI-cars       | 76.5 | 76.4 | -    | 90.3 | 2,685   | 407     | 250   | 280    |
| KITTI-pedestrian | 54.7 | 59.1 | -    | 65.1 | 6,422   | 1,443   | 204   | 609    |
| DanceTrack       | 55.1 | 38.0 | 54.2 | 89.4 | 114,107 | 139,083 | 1,992 | 3,838  |
| CroHD HeadTrack  | 44.1 | -    | 62.9 | 67.9 | 102,050 | 164,090 | 4,243 | 10,122 |

* Results are from reusing detections of previous methods and shared hyper-parameters. Tune the implementation adaptive to datasets may get higher performance.
* The inference speed is ~28FPS by a RTX 2080Ti GPU. If the detections are provided, the inference speed of OC-SORT association is 700FPS by a i9-3.0GHz CPU.
* A sample from DanceTrack-test set is as below and more visualizatiosn are available on [Google Drive](https://drive.google.com/drive/folders/1-T4jhHwhOAp42DGJ115yMlC7CkB-PNxy?usp=sharing)

    ![](assets/dancetrack0088_slow.gif)



## Get Started
* See [INSTALL.md](./docs/INSTALL.md) for instructions of installing required components.

* See [GET_STARTED.md](./docs/GET_STARTED.md) for how to get started with OC-SORT.

* See [DEPLOY.md](./docs/DEPLOY.md) for deployment support over ONNX, TensorRT and ncnn.



## Model Zoo
The pretrained model trained on DanceTrack-train set is the original version provided in [DanceTrack](https://github.com/DanceTrack/DanceTrack/tree/main/ByteTrack), we rename and redirect it on [Google Drive](https://drive.google.com/drive/folders/1LnhZVJlpufUnWuObZASIN1KwfhuvT_a8?usp=sharing) for convenience. The weights should output results as below:

| Dataset         | HOTA | IDF1 | AssA | MOTA | DetA |
| --------------- | ---- | ---- | ---- | ---- | ---- |
| DanceTrack-val  | 52.1 | 51.6 | 35.3 | 87.3 | 77.2 |
| DanceTrack-test | 55.1 | 54.2 | 38.0 | 89.4 | 80.3 |


* For model weights for other datasets, please refer to the model zoo of [ByteTrack](https://github.com/ifzhang/ByteTrack) for options.


## Demo
To run the tracker on a provided demo video from [Youtube](https://www.youtube.com/watch?v=qv6gl4h0dvg):

```shell
python3 tools/demo_track.py --demo_type video -f exps/example/mot/yolox_dancetrack_test.py -c pretrained/bytetrack_dance_model.pth.tar --path videos/dance_demo.mp4 --fp16 --fuse --save_result --out_path demo_out.mp4
```

<center>
<img src="assets/dance_demo.gif" width="600"/>
</center>


## Roadmap
We are still actively updating OC-SORT. We always welcome contributions to make it better for the community. We have some high-priorty to-dos as below:
- [x] Add more asssocitaion cost choices: GIoU, CIoU, etc.
- [ ] Support OC-SORT in [mmtracking](https://github.com/open-mmlab/mmtracking).
- [ ] Add more deployment options and improve the inference speed.
- [ ] Make OC-SORT adaptive to customized detector.


## Acknowledgement and Citation
The codebase is built highly upon [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [filterpy](https://github.com/rlabbe/filterpy), and [ByteTrack](https://github.com/ifzhang/ByteTrack). We thank their wondeful works. OC-SORT, filterpy and ByteTrack are available under MIT License. And [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) uses Apache License 2.0 License.

If you find this work useful, please consider to cite our paper:
```
@article{cao2022observation,
  title={Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking},
  author={Cao, Jinkun and Weng, Xinshuo and Khirodkar, Rawal and Pang, Jiangmiao and Kitani, Kris},
  journal={arXiv preprint arXiv:2203.14360},
  year={2022}
}
```
