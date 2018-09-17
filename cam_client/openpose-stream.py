#! /usr/bin/env python

import sys
import cv2
import json
import numpy as np
import scipy.spatial.distance as sci_dist
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
params["net_resolution"] = "-1x80"  # ~128 for MPI_4, 64 for COCO
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


class BodyPart(object):

    def __init__(self, p_id):
        self.p_id = p_id
        self.x_buf = deque([0, 0], maxlen=10)
        self.y_buf = deque([0, 0], maxlen=10)
        self.x_filt = FirstOrder(0.3)
        self.y_filt = FirstOrder(0.3)
        self.c = 0.

    def update(self, pts):
        self.x_buf.append(self.x_filt.filter(pts[0]))
        self.y_buf.append(self.y_filt.filter(pts[1]))
        self.c = pts[2]


class FirstOrder(object):

    def __init__(self, c):
        self.coeff = c
        self.last = 0.

    def filter(self, val):
        self.last = self.coeff * self.last + (1 - self.coeff) * val
        return self.last


# def create_map(pts, body_parts, name_list):
#     comp_mat = np.zeros(shape=(len(pts), len(body_parts)))
#     try_config = dict()
#     unmarked_pts = list()

#     for j in range(len(body_parts)):
#         for i in range(len(pts)):
#             x_eul = body_parts[j].x_buf[-1] + \
#                     0.2*(body_parts[j].x_buf[-1] - body_parts[j].x_buf[-2])
#             y_eul = body_parts[j].y_buf[-1] + \
#                     0.2*(body_parts[j].y_buf[-1] - body_parts[j].y_buf[-2])

#             comp_mat[i][j] = sci_dist.euclidean(
#                 np.array([x_eul, y_eul]), pts[i, :2])

#     for i in range(len(comp_mat[:, 0])):
#         if i not in try_config.keys():
#             try_config[i] = np.argmax(comp_mat[i, :])

#     return try_config

    # if len(pts) < len(body_parts):
    #     for i in range(len(body_parts) - len(pts)):
    #         name_list.append(body_parts.pop().p_id)

    # if len(pts) > len(body_parts):
    #     for i in range(len(pts) - len(body_parts)):
    #         name = name_list.pop()
    #         new_bp = BodyPart(name)

    #         body_parts.append(new_bp)

    #     print(body_parts)


def update_basic(pts1, pts2, pts3, bp1, bp2, bp3, name_list):
    for j in range(len(bp1)):

        if j >= len(pts1[:, 0]):
            reclaim = bp1.pop().p_id
            bp2.pop()
            bp3.pop()

            name_list.append(reclaim)
        else:
            bp1[j].update(pts1[j, :])
            bp2[j].update(pts2[j, :])
            bp3[j].update(pts3[j, :])

    if len(pts1) > len(bp1):
        for i in range(len(bp1), len(pts1)):
            name = name_list.pop()
            bp1.append(BodyPart(name))
            bp2.append(BodyPart(name))
            bp3.append(BodyPart(name))


video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
video_capture.set(cv2.CAP_PROP_EXPOSURE, 10)

# p_id: str,
# x: deque, y: deque,
# x_filt: fo, y_filt: fo
# c: float
# head = BodyPart('Bob')
# left_hand = BodyPart('Bob')
# right_hand = BodyPart('Bob')

heads = list()
left_hands = list()
right_hands = list()

# last_num_people = 0

name_list = ['ashley', 'bill', 'marie', 'billybob', 'katie', 'robert', 'joe',
             'tom', 'noah', 'sarah', 'sara']

while True:
    # Read new image
    ret, img = video_capture.read()
    # Output keypoints and the image with the human skeleton blended on it
    kpts, output_image = pose_net.forward(img, display=True)
    # print(head.x_buf[-1], head.y_buf[-1])

    # If there are people detected
    if kpts.shape != (0, 0, 0):

        update_basic(kpts[:, HEAD_I, :], kpts[:, LEFT_I, :], kpts[:, RIGHT_I, :],
                     heads, left_hands, right_hands, name_list)

        # for i in range(len(kpts[:, 0, 0])):

        #     head.update(kpts[i, HEAD_I, :])
        #     left_hand.update(kpts[i, LEFT_I, :])
        #     right_hand.update(kpts[i, RIGHT_I, :])

        for bp in heads:  # , left_hand, right_hand]:
            for l in range(len(bp.x_buf)-1):
                cv2.line(output_image, (int(bp.x_buf[l]), int(bp.y_buf[l])),
                         (int(bp.x_buf[l+1]), int(bp.y_buf[l+1])), (0, 255, 0),
                         int(5*bp.c))

        # last_num_people = len(kpts[:, 0, 0])

        package = list()
        for i in range(len(heads)):
            per_dict = dict()
            per_dict['person_id'] = heads[i].p_id
            per_dict['x'] = 100 - (100*(heads[i].x_buf[-1]/1280))
            per_dict['y'] = 50*(heads[i].y_buf[-1]/720) + 10
            if left_hands[i].x_buf[-1] > 0:
                per_dict['hand1_x'] = 110 - (120*(left_hands[i].x_buf[-1]/1280))
                per_dict['hand1_y'] = 100*(left_hands[i].y_buf[-1]/720) + 40
            if right_hands[i].x_buf[-1] > 0:
                per_dict['hand2_x'] = 110 - (120*(right_hands[i].x_buf[-1]/1280))
                per_dict['hand2_y'] = 100*(right_hands[i].y_buf[-1]/720) + 40
            package.append(per_dict)

        print package

        # Send to server
        sock_client.emit('messages', json.dumps(package))

    # Display the image
    cv2.imshow("output", output_image)
    cv2.waitKey(2)
