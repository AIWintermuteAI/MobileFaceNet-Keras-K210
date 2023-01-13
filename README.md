# MobileFaceNet-Keras
A Keras implementation of MobileFaceNet from [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573). Modified to be converted to K210.

NOTE: Code is archived and not maintained. Tested with CASIA dataset.
Below is the original readme. Original repository: https://github.com/LeonFaith/MobileFaceNet-Keras

2020.03.27: Modify the implementation of ArcFace loss function and revise the training codes for TensorFlow 2 tf.keras API.
2020.03.30: Modify the implementation of resume training scripts for ImageDataGenerator and add the training codes for TensorFlow 2 tf.data API with TFRecords. Thanks to TensorFlow 2, now the training speed on my computer is almost 4 times faster when either using ImageDataGenerator API or TFRecord with tf.data API!

## 1. Data Preprocessing Strategy
(1) Use the celebrity & msra datasets from the Trillion Pairs dataset: http://trillionpairs.deepglint.com/data.
(2) Set a standard face criterion to align the face & then crop the 112x112 area.
(3) For each identity folder:
a. n > 350
Randomly pick 350 pics from the origin data set
b. 200 < n <= 350
Keep all the pics
c. 90 < n <= 200
Keep all the pics & Apply Opening to them (Double the data)
d. 30 < n <= 90
Keep all the pics, Apply Opening to them, Add Gaussian noise to them & Add Salt & Pepper noise to them (Four times the data)
e. n <= 30
Drop the folder
(4) PS: The OpenCV ops in Python run really slow and all the operations above have a quite similar code implementation in C++. Thus it is recommended to use C++ for prerocessing the image dataset (OpenCV supports reading the Caffe version of MTCNN), as well as to use multi-thread processing.

## 2. Training Strategy
(1) Train the model with SoftMax loss for pre-training.
(2) After the loss becomes lingering around some value, change the loss to ArcFace loss and resume training.
The GPU memory is not enough for mini-batch size 512 (as did in the original paper) for training on a Nvidia 2080 Ti, thus I have to downsize it to 128 to fit in the memory.

## 3. Improvement for training step in progress.
The training data have been finished augmentation. There are 15,090,270 pics of 67,960 identities in the set and I choose 0.005 out of the data for validation during training. Now the ArcFace loss has been modified and experimented to be functioning right.
To-do list:
Now the multiple TFRecord files are being streamed as a sequence with their name orders and the buffer size is smaller than the total size of image set because of the limit of GPU memory. For exactly randomizing the order of images when being trained, the input pipeline should be improved further.

## References
(1) Original paper of MobileFaceNet: [MobileFaceNet](https://arxiv.org/abs/1804.07573)
(2) The idea of the implementation of model structure is from MobileNet v2: [xiaochus/MobileNetV2](https://github.com/xiaochus/MobileNetV2)
(3) The implementation of ArcFace loss (InsightFace loss) is from: [ewrfcas/ArcFace_loss](https://github.com/ewrfcas/Machine-Learning-Toolbox/blob/127d6e5d336614d1efb21e78865501435cdb7b8b/loss_function/ArcFace_loss.py)
(4) MTCNN library (the best MTCNN I've ever used): [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
(5) The idea of cropping face is from: [deepinsight/insightface/face_preprocess](https://github.com/deepinsight/insightface/blob/master/src/common/face_preprocess.py)
