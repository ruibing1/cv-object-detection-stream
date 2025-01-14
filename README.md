
# cv-object-detection-stream

This project focuses on real-time object detection in (YouTube) video streams, leveraging technologies such as Python 3.6, OpenCV with YOLOv3 detection method, and Streamlink.

The configurations include the following:
* A Python environment
`$ pip install numpy pandas argparse datetime pafy streamlink flask_opencv_streamer youtube-dl flask_opencv_streamer opencv-contrib-python`
* An installation of OpenCV with YOLOv3 (https://pjreddie.com/darknet/yolo/) 
* An instance of Streamlink (https://github.com/streamlink/streamlink)

The main function of the project is to detect objects in a video stream and write the number of detected objects to an output file every x seconds (defaults to 5 seconds).

The starting point of this repository is the source code from the blog: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/.

## YOLO Weights:
The YOLOv3 weights can be downloaded using the command: `$ wget https://pjreddie.com/media/files/yolov3.weights` 
The weights have been trained on the COCO dataset (http://cocodataset.org)

## Live stream object detection example:
Example streams: Jackson Hole Wyoming USA Live Cams - SeeJH.com
* https://youtu.be/1EiC9bvVGnk
* https://youtu.be/RZWzyQuFxgE

The example was accomplished with confidence level 0.30 and threshold level 0.55.

Screenshots of the sample output:

![](JacksonHole1.jpg)
![](JacksonHole2.jpg)

This project is maintained by ruibing1.