import cv2
import numpy as np
import tensorflow as tf
cap = cv2.VideoCapture(0)
new_model = tf.keras.models.load_model('SV.model')

a = 130

def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out

while True:
	ret, img = cap.read()
	img = cv2.flip(img, 1) 
	x=350
	y=50
	h=250
	w=250
	cv2.rectangle( img, (x,y) , (x+w,y+h) ,(255,0,0),2)
	gray = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
	_,gray = cv2.threshold(gray,a,255,cv2.THRESH_BINARY)

	#gray = cv2.Canny(gray,100,200)
	
	gray = fillHole(gray)
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
	ans = np.argmax(prediction)
	# Using cv2.putText() method 
	ans = 65+int(ans)
	ans = chr(ans)
	img = cv2.putText(img,ans, org, font,fontScale, color, thickness, cv2.LINE_AA) 
	   
	
	
	cv2.imshow('img',img)
	
	k = cv2.waitKey(30) & 0xff
	if k==27:
		break;
	if k==50:
		a=a+1;
	if k==56:
		a=a-1

cap.release()
cv2.destroyAllWindows()