import cv2
import numpy as np
import os
import handshape_feature_extractor 
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

# import torch
# import torch.nn as nn

middle_frames = []
extractor = HandShapeFeatureExtractor()

with open('train_frames.csv', 'r') as read_obj:
	csv_reader = reader(read_obj)
	for row in csv_reader:
		row = np.array([float(x) for x in row])
		middle_frames.append(row)
middle_frames = np.array(middle_frames)


outputs = []
with open('train_outputs.csv', 'r') as read_obj:
	csv_reader = reader(read_obj)
	for row in csv_reader:
		outputs.append(int(row[0]))
outputs = np.array(outputs)

results = []

gnames = os.listdir("test")
for gname in gnames:
	path = os.path.join("test",gname)
	frames = frameExtractor(path)
	fvect = []
	for frame in frames:
		fv = extractor.extract_feature(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
		# fvect.append(np.array(fv).argmax(0))
		fvect.extend(fv)
	# for frame in frames:
	# 	fvect.extend(extractor.extract_feature(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0])
	cc=0
	mindisti, mindist = 0,1000
	for v in middle_frames:
		cosdist = spatial.distance.cosine(fvect,v)
		if cosdist<mindist:
			mindist = cosdist
			mindisti = cc
		cc+=1
	
	results.append([ outputs[mindisti] ])
	# results.append([ svc.predict([fvect])[0] ])
filename = "results.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(results)