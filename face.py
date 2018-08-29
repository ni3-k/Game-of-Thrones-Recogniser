import cv2
import pickle 
import numpy as np

cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("training.yml")
label_id = pickle.load(open("labels.pickle", "rb"))
label_id = {v: k for k,v in label_id.items()}

def detect_faces(cascade, image, scaleFactor = 1.2):
	# Making a copy of image
	image_copy = image.copy()          
	
	# Converting to gray image
	gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)         

	# Detect faces using OpenCV Cascades
	faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          

	# Plotting rectangle over faces
	for (x, y, w, h) in faces:
		cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              

	if len(faces) > 0:
		return image_copy, gray[y:y+h, x:x+w]
	else:
		return image_copy, None

def draw_text(img, text, x=400, y=300):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#video = cv2.VideoCapture(0)
url = "http://192.168.42.129:8080/shot.jpg"
import requests

while True:
	imgResp=requests.get(url)
	imgNp=np.array(bytearray(imgResp.content),dtype=np.uint8)
	frame=cv2.imdecode(imgNp,-1)

	#ret, frame = video.read()
	face_detected, roi = detect_faces(cascade, frame)
	if roi is not None:
		id_, conf = face_recognizer.predict(roi)
		print(label_id[id_], conf)
		draw_text(face_detected, label_id[id_])
	cv2.imshow('face', face_detected)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 
#video.release()
cv2.destroyAllWindows()
