# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import os
import handshape_feature_extractor 
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

middle_frames = []
extractor = HandShapeFeatureExtractor().get_instance()

## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

gnames = sorted(os.listdir("traindata/"))
outputs = []
c=0
gestID = {}
for x in range(10):
	gestID["Num"+str(x)] = x
	# gestID[str(x)] = x
gestID["FanDown"] = 10
gestID["FanOn"] = 11
gestID["FanOff"] = 12
gestID["FanUp"] = 13
gestID["LightOff"] = 14
gestID["LightOn"] = 15
gestID["SetThermo"] = 16
for gname in gnames:
	path = os.path.join("traindata",gname)
	impath = frameExtractor(path,"trainframes",c)
	img = cv2.imread(impath)
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fvect = extractor.extract_feature(frame).squeeze()
	middle_frames.append(fvect)
	# gn = gname.split("_")[0]
	outputs.append(c%17)
	c+=1

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

results = []
gnames = os.listdir("test/")
c=0
for gname in gnames:
	path = os.path.join("test",gname)
	impath = frameExtractor(path,"testframes",c)
	img = cv2.imread(impath)
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fvect = extractor.extract_feature(frame).squeeze()
	cc=0
	mindisti, mindist = 0,1e8
	for v in middle_frames:
		cosdist = spatial.distance.cosine(fvect,v)
		if cosdist<mindist:
			mindist = cosdist
			mindisti = cc
		cc+=1
	print(gname, outputs[mindisti])
	results.append([ outputs[mindisti] ])
	c+=1


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

filename = "Results.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(results)