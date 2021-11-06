import cv2
import numpy as np
import os
import handshape_feature_extractor 
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

import torch
import torch.nn as nn

middle_frames = []
extractor = HandShapeFeatureExtractor()

middle_frames = []
outputs = []
gnames = sorted(os.listdir("traindata"))
c=0
for gname in gnames:
	path = "traindata/"+gname
	frames = frameExtractor(path)
	fvect=[]
	for frame in frames:
		frame = cv2.ROTATE_90_CLOCKWISE(frame, cv2.ROTATE_90_CLOCKWISE)
		fvect.extend(extractor.extract_feature(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0])
	outputs.append(c)
	c+=1
	middle_frames.append(np.array(fvect))

middle_frames = np.array(middle_frames)
outputs = np.array(outputs)

results = []

gnames = os.listdir("test")
for gname in gnames:
	path = os.path.join("test",gname)
	frames = frameExtractor(path)
	fvect = []
	for frame in frames:
		frame = cv2.ROTATE_90_CLOCKWISE(frame, cv2.ROTATE_90_CLOCKWISE)
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
	# t_vect = torch.from_numpy(np.array(fvect)).unsqueeze(0).double().cuda()
	# pop = fc(t_vect).argmax(1)
	# print(gname, pop[0].item())
	# results.append([ pop[0].item() ])
	results.append([ outputs[mindisti] ])
	# print(gname, outputs[mindisti])
	# results.append([ svc.predict([fvect])[0] ])
filename = "results.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(results)