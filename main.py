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
'''
with open('train_frames.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		middle_frames.append([float(x) for x in row])

with open('train_outputs.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		outputs.append(int(row[0]))

'''
for gname in gnames:
	path = os.path.join("traindata",gname)
	frames = frameExtractor(path,0)
	for frame in frames:
		fvect=[]
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fvect.extend(np.squeeze(extractor.extract_feature(frame)))
		middle_frames.append(fvect)
		outputs.append(c%17)
	c+=1

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

results = []
gnames = sorted(os.listdir("test/"))
c=0

for gname in gnames:
	path = os.path.join("test",gname)
	frames = frameExtractor(path,-1)
	fvect=[]
	for frame in frames:
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fvect.extend(np.squeeze(extractor.extract_feature(frame)))
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