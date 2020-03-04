# Unsupervised Learning of Depth, Optical Flow and Pose with Occlusion from 3D Geometry

Code for the papers: 

*G. Wang, H. Wang, Y. Liu, and W. Chen,*  [**Unsupervised Learning of Monocular Depth and Ego-Motion Using Multiple Masks**](https://ieeexplore.ieee.org/abstract/document/8793622), in International Conference on Robotics and Automation, pp. 4724-4730, 2019.

*G. Wang, C. Zhang, H. Wang, J. Wang, Y. Wang, and X. Wang,*  [**Unsupervised Learning of Depth, Optical Flow and Pose with Occlusion from 3D Geometry**](https://arxiv.org/abs/2003.00766), under review.

## Prerequisites

Python3 and pytorch are required. Besides, other libraries need to be installed by runing:
```
pip3 install -r requirements.txt
```

## Preparing training data

#### KITTI
For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command.

```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 832 --height 256 --num-threads 1 --static-frames data/static_frames.txt --with-gt
```

For testing optical flow ground truths on KITTI, download [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset. You need to download 1) `stereo 2015/flow 2015/scene flow 2015` data set (2 GB), 2) `multi-view extension` (14 GB), and 3) `calibration files` (1 MB) . You should have the following directory structure:
```
kitti2015
  | data_scene_flow  
  | data_scene_flow_calib
  | data_scene_flow_multiview  
```

#### Cityscapes

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it.

```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ --dataset-format 'cityscapes' --dump-root /path/to/resulting/formatted/data/ --width 832 --height 342 --num-threads 1
```

Notice that for Cityscapes the `img_height` is set to 342 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 256.

## Training

```
python3 train.py /path/to/prepared/data \
--dispnet DispResNetS6 --posenet PoseNetB6 --flownet Back2Future \
-b 4 -pc 1.0 -pf 0.0 -m 0.0 -c 0.0 -s 0.2 \
--epoch-size 100 --log-output -f 30 --nlevels 6 --lr 1e-4 -wssim 0.85 --epochs 4000 \
--smoothness-type edgeaware --fix-masknet --fix-flownet --with-depth-gt --log-terminal \
--spatial-normalize-max --workers 8 --kitti-dir /data/to/kitti --add-less-than-mean-mask \
--add-maskp01 --using-none-mask --name demo \
--pretrained-disp /path/to/disp/model \
--pretrained-pose /path/to/pose/model
```

Tensorboard can be open with the command:
```
tensorboard --logdir=./
```
and visualize the training progress by opening https://localhost:6006 on your browser.

## Evaluation

#### Disparity

```
python3 test_disp.py --dispnet DispResNetS6 --pretrained-dispnet /path/to/dispnet --pretrained-posent /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list
```

#### Pose

```
python test_pose.py pretrained/pose_model_best.pth.tar --img-width 832 --img-height 256 --dataset-dir /path/to/kitti/odometry/ --sequences 09 --posenet PoseNetB6
```


#### Optical Flow

```
python test_flow.py --pretrained-disp /path/to/dispnet --pretrained-pose /path/to/posenet --pretrained-mask /path/to/masknet --pretrained-flow /path/to/flownet --kitti-dir /path/to/kitti2015/dataset
```

## Downloads
#### Pretrained Models
- [DispNet, PoseNet, and FlowNet](https://jbox.sjtu.edu.cn/l/6uq1SX) in joint unsupervised learning of depth, pose and optical flow.


## Acknowlegements
We are grateful to Anurag Ranjan for his [github repository](https://github.com/anuragranj/cc). Our code is based on theirs. 

## References

*G. Wang, H. Wang, Y. Liu, and W. Chen,* **Unsupervised Learning of Monocular Depth and Ego-Motion Using Multiple Masks**, in International Conference on Robotics and Automation, pp. 4724-4730, 2019.

*G. Wang, C. Zhang, H. Wang, J. Wang, Y. Wang, and X. Wang,*  **Unsupervised Learning of Depth, Optical Flow and Pose with Occlusion from 3D Geometry**, under review.
