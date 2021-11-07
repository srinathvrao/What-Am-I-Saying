# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import csv
import numpy as np
import os
from frameextractor import frameExtractor
import handshape_feature_extractor 
from handshape_feature_extractor import HandShapeFeatureExtractor

middle_frames = []
outputs = []

gestID = {}
for x in range(10):
	gestID["Num"+str(x)] = x
gestID["FanDown"] = 10
gestID["FanOn"] = 11
gestID["FanOff"] = 12
gestID["FanUp"] = 13
gestID["LightOff"] = 14
gestID["LightOn"] = 15
gestID["SetThermo"] = 16

test_middle_frames = []
test_outputs = []

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)

cv2.waitKey(0)
cv2.destroyAllWindows()

extractor = HandShapeFeatureExtractor()
# print(extractor.model.summary())

gnames = os.listdir("traindata")
for gname in gnames:
	print(gname)
	gnameids = os.listdir("traindata/"+gname)
	# for x in range(2):
	for gnameid in gnameids:
		path = "traindata/"+gname+"/"+gnameid #str(x)+"_Venkobarao.mp4"
		frames = frameExtractor(path,"fpath",0)
		for frame in frames:
			fvect=[]
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			frame = adjust_gamma(frame, gamma=0.4)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			fvect.extend(np.squeeze(extractor.extract_feature(frame)))
			middle_frames.append(np.array(fvect))
			outputs.append([gestID[gname]])
		print(len(frames), len(fvect))
	# print("=====")

middle_frames = np.array(middle_frames)
outputs = np.array(outputs)
filename = "train_frames.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(middle_frames)

filename = "train_outputs.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(outputs)

print(middle_frames.shape, outputs.shape)