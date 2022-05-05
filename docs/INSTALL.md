## Installation
OC-SORT is built upon codebase of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and [ByteTrack](https://github.com/ifzhang/ByteTrack). I tested the code with Python 3.8. 

### 1. Installing on the host machine
Step1. Install OC-SORT
```shell
git clone https://github.com/noahcao/OC_SORT.git
cd OC_SORT
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox pandas xmltodict
```
### 2. Docker build
```shell
docker build -t ocsort:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/OC_SORT/pretrained \
-v $PWD/datasets:/workspace/OC_SORT/datasets \
-v $PWD/YOLOX_outputs:/workspace/OC_SORT/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
ocsort:latest
```
