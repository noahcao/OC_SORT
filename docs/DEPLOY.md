# Deployment 

We provide support to some popular deployment tools. This part is built upon the implementation of [YOLOX Deployment](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo) and [the adaptation by ByteTrack](https://github.com/ifzhang/ByteTrack/tree/main/deploy).


## ONNX support 

1. convert the pytorch model to onnx checkpoints, we provide an example here. 
    ```python
    # In pratice you may want smaller model for faster inference.
    python deploy/scripts/export_onnx.py --output-name  ocsort.onnx -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar
    ```

2. run on the provided model video by
    ```shell
    cd $OCSORT_HOME/deploy/ONNXRuntime
    python onnx_inference.py
    ```

## TensorRT support (Python)

1. Follow [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) and [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to install TensorRT (Version 7 recommended) and torch2trt.

2. Convert Model
    ```python
    # you have to download checkpoint bytetrack_s_mot17.pth.tar from model zoo of ByteTrack
    python3 deploy/scripts/trt.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar
    ```

3. Run on a demo video
    ```python
    python3 tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py --trt --save_result
    ```

*Note: We haven't validated the C++ support for TensorRT yet, please refer to [ByteTrack guidance](https://github.com/ifzhang/ByteTrack/tree/main/deploy/TensorRT/cpp) for adaptation for now.*

## ncnn support
Please follow the [guidelines](https://github.com/ifzhang/ByteTrack/tree/main/deploy/ncnn/cpp) from ByteTrack to deploy by support from ncnn.