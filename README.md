# unet-rgb

Keras(TF backend) implementation of unet for RGB pictures.

---

## Overview
### Requirement
- OpenCV
- Python 2.7
- Tensorflow-gpu
- Keras

### Data
You can download all data here:https://pan.baidu.com/s/1WqmR-9jodyyUnLRCK6pcVw password: 533h

CamVid Data:https://github.com/preddy5/segnet/tree/master/CamVid

### Howtouse
- extract downloaded data to corresponding directories
- run ```python data.py``` to generate 3 .npy files or you can download them to npydata
- run ```python unet.py``` to train, you can change hyperparameters on your own situation
- run ```python test2mask2pic.py``` to get test results(pictures)

### Results
After training about 30 epochs, loss goes to about 0.005.The results seems OK! But the edges look a bit rough. I think that is Unet's own limitation.

![img/0test.jpg](img/0test.jpg)

![img/0label.jpg](img/0label.jpg)

### Camvid
- run ```python data_camvid.py```
- run ```python unet_camvid.py```
- run ```python predict_camvid.py```

After 50 epochs training, loss goes to about 4e-04 and acc is 0.9644.

![img/1test.png](img/1camvid.png)
![img/1label.png](img/1label.png)

## About
Unet is mostly used in medical areas. I used this model for semantic segmentation of satellite remote sensing images in real work and the result is not bad. I think one of the reason is that it's a coarse-grained task like medical image analysis. Of course, enough labeled images are necessary.

## Reference
https://github.com/zhixuhao/unet
