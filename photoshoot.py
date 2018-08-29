import cv2
import os
import numpy as np

name = input("Name:")
name = name.replace(" ", "-").lower()
print(name)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "data", name)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

url = "http://192.168.42.129:8080/shot.jpg"
import requests
start = 1

while True:
	imgResp=requests.get(url)
	imgNp=np.array(bytearray(imgResp.content),dtype=np.uint8)
	frame=cv2.imdecode(imgNp,-1)

	if cv2.waitKey(1) & 0xFF == ord('n'):
		path = os.path.join(image_dir, str(start)+".jpg")
		print(path)
		cv2.imwrite(path, frame)
		start += 1
		print(start)

	cv2.imshow('face', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 
#video.release()
cv2.destroyAllWindows()

