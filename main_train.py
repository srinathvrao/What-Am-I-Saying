# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
from frameextractor import frameExtractor
import cv2
import csv
import numpy as np
import os
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
		fvect=[]
		for frame in frames:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			fvect.extend(np.squeeze(extractor.extract_feature(frame)))
		outputs.append([gestID[gname]])
		middle_frames.append(np.array(fvect))
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