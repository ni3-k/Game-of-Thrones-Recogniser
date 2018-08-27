import os
import cv2
import numpy as np
import faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data")

haar_face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".png"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			image = cv2.imread(path)
