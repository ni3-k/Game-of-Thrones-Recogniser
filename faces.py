import numpy as np
import cv2


def detect_faces(image, cascade, scaleFactor=1.2, roi=False):
	image_copy = image.copy()
	gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

	faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor)

	if roi is False:
		for (x, y, w, h) in faces:
			cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return image_copy
	else:
		face_array = []
		for (x, y, w, h) in faces:
			face = gray[y:y+h, x:x+w]
			face_array.append(face)
		return face_array

