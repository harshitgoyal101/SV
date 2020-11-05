import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "gestures"
CATEGORIES = 45


training_data = []

for category in range(CATEGORIES):
	try:
		path = os.path.join(DATADIR,str(category))
		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
			new_array = cv2.resize(img_array,(50,50))
			training_data.append([new_array,category])
	except Exception as e:
		pass

X = []
y = []

for features,label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1,50,50,1)

import pickle
pickle_out = open("X.pickle",'wb')
pickle.dump(X,pickle_out)
pickle_out.close()


pickle_out = open("y.pickle",'wb')
pickle.dump(y,pickle_out)
pickle_out.close()