# Object_detect_YOLO V3 and V4

[![](https://img.shields.io/badge/NVIDIA-GTX1050-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-downloads)
[![](https://img.shields.io/badge/Ubuntu18.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)]()
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>


Installation of CUDA and its requirement to run YOLOv3-v4

## (Neural Networks for object detection)
---
YOLO: You only look once (YOLO) is a state-of-the-art, real-time object detection system.

Get detections on a single pass.
1. Resize the input image (e.g., 448x448).
2. Run Convolutional Network to get detections.
3. Filter the output by a Non-max supression algorithm (to remove multiple detections of the same object)


Architecture (training phase):

 -   Input size: 448 x 448
 -   Nine Convolutional layers with leaky ReLU.
 -   Six max pooling layers.
-  Backbone feature map: 7 x 7 (7 x 64 = 448)

Output size: 30 x 7 x 7:

- Two bounding box definitions, containing x, y, height, width andconfidence. (2 x 5)
- Class probabilities, relevant if only object confidence is high enough. (20)


**Box probabilities:**

If there is an object in a cell, we decide which object by finding the biggest probability.

**Non-Max Supreesion:**

- An object may be detected several times by different boxes (grid cells).
- So, we need to use non-max supression to filter the results.
- Final detections are obtained by using thresholding and non-max supression.

![](https://github.com/Foroozani/Object_detect_YOLO3-4/blob/main/images/FilterResults.png)




For more information see the [Darknet: Open Source Neural Networks in C](https://pjreddie.com/darknet/) 

source code:  https://github.com/AlexeyAB/darknet



# Description
Here, I guide you step by step with a bare machine to get a real time object detector with YOLO v3 and V4.

## Step-1. Prepare machine and environment on Linux

 Installation Summary:
   * Install Ubuntu 18.03
   * Install Git
   * Install Cmake >= 3.12
   * Install CUDA 10.2 (Nvidia GPU only!)
   * Install cuDNN 7.6.4 (compatible for CUDA 10.1)
   * Install Python3 and Pip3
   * Install OpenCV 3.2.0
   * Install GC

A `Ubuntu 18.04` native system is preferred in training process. At least one NVIDIA GPU Card is required such as GeForce series to enable GPU mode. One can train a network with custom data but for a videa you will end up killing the process as it may take days. 

I recently tested running YOLo on both 12 CPU and GPU. Here are the steps you may follow:

**(a) Environment**
Create a virtual environment and start installing dependencies and libraries.

**(b) Requirements** 

```bash 
# Install Git
apt-get install git
git --version

# Install Cmake >= 3.12
apt install -y cmake
cmake --version
```
**Install CUDA**
NOTE: YOLO Darknet from AlexeyAB repository can be used with CUDA 10.2. To install `cuda` you can also visit [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive). On terminal shell you can follow these steps:

```bash 
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb

sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install cuda

#NOTE: Restart Ubuntu after CUDA installed!!
```

Then set-up CUDA 10.2 path 

```bash 
echo "export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}" >> ~/.bashrc
source ~/.bashrc
```
To verify CUDA is installed successfully, run the comand `ncc --version`, and `nvidia-sim.

**Install cuDNN**

Next, install cuDNN 7.6.4 (compatible for CUDA 10.1) [see also](https://www.youtube.com/watch?v=UhuK9ShIpf8). You can download [cuDNN Runtime Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/rdp/cudnn-archive), [cuDNN Developer Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/rdp/cudnn-archive), [cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)]( https://developer.nvidia.com/rdp/cudnn-archive) also these three libraries manually and save in the same folder and run the command:

```bash 
sudo dpkg -i libcudnn7_7.6.4.38-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.1_amd64.deb
```

To verify that cuDNN is installed and is running properly, compile the `mnistCUDNN` sample located in the `/usr/src/cudnn_samples_v7` directory in the Debian file [see](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify).

**Procedure**
```bash 
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```


If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:
*Test passed!*


Next, we can upgrade `pip` and install `OpenCV`, 

```bash 
sudo python -m pip install --upgrade pip
sudo pip3 install opencv-python

# Verify installation
# python (enter python shell)
# >> import cv2
# >> print (cv2.__version__)
```
Also, one can check the gcc version with command `gcc ---version`. 

# Step-2. Download this repo

For YOLO V3, I follow the steps from here, [YOLO V3](https://pjreddie.com/darknet/yolo/) and run a test image as shown here. 

```bash 
# Install Git 
# sudo apt-get install git
git clone https://github.com/pjreddie/darknet
cd darknet
make

#download the pre-trained weight file
wget https://pjreddie.com/media/files/yolov3.weights 

#Then run the detector!
./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg

#Easy!Done, You can make a your own image 
```
![](https://github.com/Foroozani/Object_detect_YOLO3-4/blob/main/images/predictions_v3.jpg)


Here is repositories for **YOLO V4** [darknet](https://github.com/AlexeyAB/darknet). 
```bash 
git clone https://github.com/AlexeyAB/darknet.git
```
Download `yolov4.weights` file 245 MB (https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights). 

Before, you compile, make few changes in `Makefile` if you wish to run on GPU `GPU=1, CUDNN=1, CUDNN_HALF=1`. Otherwise you can unable it. (see make files). Then run the command `make clean` and then `make`. Note everytime, you do any changes you first run `make clean`. Now the executable file 'darknet' is created nd run the darknet:

```bash
./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
```

to have a video you must use GPU:

```bash 
./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights data/***.mp4
```
![YOLO-V3](https://github.com/Foroozani/Object_detect_YOLO3-4/blob/main/images/predictions_yolo4.jpg)

Here is the output of my image
data/4.JPG: Predicted in 1711.256000 milli-seconds.
zebra: 100%
person: 99%
handbag: 60%
person: 28%

Yolo-V4 is faster and nicer, is not it?

## Credits
---
Original Code: https://github.com/pjreddie/darknet

Site: https://pjreddie.com/darknet/yolo/
