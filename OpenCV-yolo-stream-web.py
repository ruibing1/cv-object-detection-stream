
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
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("Initializing...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

url = args["url"]

vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
streams = streamlink.streams(url)

# set initial parameters
writer = None
(W, H) = (None, None)
starttime=time.time()
frame_ind = 0
obj = np.zeros((1000,7))
# loop over frames from the video file stream
while True:
    # read the next frame from the file
    framedatetime = datetime.datetime.now()
    framedatetime = framedatetime.strftime('%Y%m%d%H%M%S')
    cap = cv2.VideoCapture(streams["best"].url)
    (grabbed,frame) = cap.read()
    #(grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])