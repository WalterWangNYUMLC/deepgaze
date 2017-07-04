#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2016 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf
import cv2
import os
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator


sess = tf.Session()  # Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object


dir_path = os.path.dirname(os.path.realpath(__file__))
pitchfile_path = os.path.abspath(os.path.join(dir_path, "../../etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))
yawfile_path = os.path.abspath(os.path.join(dir_path, "../../etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"))
print(yawfile_path)
my_head_pose_estimator.load_pitch_variables(pitchfile_path)
my_head_pose_estimator.load_yaw_variables(yawfile_path)
image = cv2.imread("1.jpg")  # Read the image with OpenCV

pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
print("Estimated pitch ..... " + str(pitch[0,0,0]))
print("Estimated yaw ..... " + str(yaw[0,0,0]))
