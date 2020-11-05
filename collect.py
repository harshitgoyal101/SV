import cv2
import numpy as np
import tensorflow as tf
cap = cv2.VideoCapture(0)
new_model = tf.keras.models.load_model('SV.model')


while True:
	ret, img = cap.read()
	img = cv2.flip(img, 1) 
	x=350
	y=50
	h=250
	w=250
	cv2.rectangle( img, (x,y) , (x+w,y+h) ,(255,0,0),2)

	hsvim = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2HSV)
	lower = np.array([0, 48, 80], dtype = "uint8")
	upper = np.array([20, 255, 255], dtype = "uint8")
	skinRegionHSV = cv2.inRange(hsvim, lower, upper)
	blurred = cv2.blur(skinRegionHSV, (2,2))
	ret,thresh = cv2.threshold(blurred,170,255,cv2.THRESH_BINARY)
	cv2.imshow("thresh", thresh)


	gray = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
	_,gray = cv2.threshold(gray,170,255,cv2.THRESH_BINARY)
	#gray = cv2.Canny(gray,100,200)
	cv2.imshow('gray',gray)
	gray = cv2.resize(gray,(50,50))
	gray = np.array(gray).reshape(-1,50,50,1)
	gray = gray/255.0
	
	prediction = new_model.predict([gray])
		# org 
	org = (50, 50) 
	  
	# fontScale 
	fontScale = 2
	font = cv2.FONT_HERSHEY_SIMPLEX 
  	   
	# Blue color in BGR 
	color = (255, 0, 0) 
	  
	# Line thickness of 2 px 
	thickness = 2
	ans = str(np.argmax(prediction))
	# Using cv2.putText() method 
	img = cv2.putText(img,ans, org, font,fontScale, color, thickness, cv2.LINE_AA) 
	   
	
	
	cv2.imshow('img',img)
	
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break;

cap.release()
cv2.destroyAllWindows()