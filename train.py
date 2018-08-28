import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data")

cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

x_train = []
y_labels = []
start_label = 0
label_id = {}

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".png"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			image = cv2.imread(path)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			
			# Detecting Face and Creating Training Data
			faces = cascade.detectMultiScale(gray, scaleFactor=1.2)
			for (x, y, w, h) in faces:
				face = gray[y:y+h, x:x+w]
				x_train.append(face)

			# Labels
			if label not in label_id:
				label_id[label] = start_label
				start_label += 1
			id = label_id[label]
			y_labels.append(id)

print(x_train, y_labels)
print(label_id)
