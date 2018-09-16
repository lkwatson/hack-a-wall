#! /usr/bin/env python

import sys
import cv2
import json
from socketIO_client_nexus import SocketIO, LoggingNamespace
import os
from sys import platform
from collections import deque

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../openpose/build/python')

from openpose import *

USE_MODEL = "MPI_4_layers"  # BODY_25, COCO, MPI, or MPI_4_layers
HEAD_I = 0
LEFT_I = 7
RIGHT_I = 4

sock_client = SocketIO('localhost', 3000, LoggingNamespace)

params = dict()
params["render_pose"] = False
params["disable_multi_thread"] = True

params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x128"  # ~128 for MPI_4, 64 for COCO
params["model_pose"] = USE_MODEL
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
params["default_model_folder"] = dir_path + "/../../../models/"

# Construct OpenPose object allocates GPU memory
pose_net = OpenPose(params)

video_capture = cv2.VideoCapture(0)

heads = dict()
left_hands = dict()
right_hands = dict()

head = deque(maxlen=10)
left_hand = deque(maxlen=10)
right_hand = deque(maxlen=10)

while True:
    # Read new image
    ret, img = video_capture.read()
    # Output keypoints and the image with the human skeleton blended on it
    kpts, output_image = pose_net.forward(img, display=True)
    # i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    # for i in range(len(kpts[:, 0, 0])):
    print (kpts)
    # If there are people detected
    if kpts.shape != (0, 0, 0):
        head.append([kpts[0, HEAD_I, j] for j in [0, 1, 2]])
        left_hand.append([kpts[0, LEFT_I, j] for j in [0, 1, 2]])
        right_hand.append([kpts[0, RIGHT_I, j] for j in [0, 1, 2]])

        for line_arr in [head, left_hand, right_hand]:
            if len(line_arr) > 1:
                for l in range(len(line_arr)-1):
                    cv2.line(output_image, (line_arr[l][0], line_arr[l][1]),
                             (line_arr[l+1][0], line_arr[l+1][1]), (0, 255, 0),
                             int(5*line_arr[l][2]))

    package = list()
    package.append(dict())
    package[0]['person_id'] = 'Bill'
    package[0]['x'] = 0
    package[0]['y'] = 1

    sock_client.emit('messages', json.dumps(package))

    # Display the image
    cv2.imshow("output", output_image)
    cv2.waitKey(2)
