
# run in command prompt (no output files)
# python OpenCV-yolo-stream-web.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk

# run in command prompt (with output files)
# python OpenCV-yolo-stream-web.py --yolo yolo-coco --url https://youtu.be/1EiC9bvVGnk --output output/ouput_videosteam.avi --data output/CSV/data_videosteam.csv 

# JacksonHole streams https://youtu.be/RZWzyQuFxgE & https://youtu.be/1EiC9bvVGnk

# import the necessary packages
import numpy as np
import pandas as pd
import argparse
import time
import datetime
import cv2
import os
import pafy
import streamlink
from flask_opencv_streamer.streamer import Streamer

# port stream settings
port = 4455
require_login = False
streamer = Streamer(port, require_login)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True,
    help="video url")
ap.add_argument("-p", "--period", type=float, default=5,
    help="execution period")
ap.add_argument("-o", "--output", required=False,
    help="path to output video")
ap.add_argument("-d", "--data", required=False,
    help="path to output csv")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to yolov weights, cfg and coco directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.55,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# set execution period 
period = args["period"]

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label