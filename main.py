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

'''
fc = nn.Sequential(
	nn.Linear(135,17),
	nn.Softmax(dim=1)
).double().cuda()
import torch.optim as optim
optimizer = optim.Adam(fc.parameters(),lr=1e-4)
celoss = nn.CrossEntropyLoss()
for param in fc.parameters():
	param.requires_grad = True
print(middle_frames.shape,outputs.shape)
for ep in range(20):
	# if ep%100==0:
	print("epoch",ep+1)
	for i in range(0,102,25):
		ft = torch.from_numpy(middle_frames[i:i+25]).cuda()
		top = torch.from_numpy(outputs[i:i+25]).cuda()
		pop = fc(ft)
		loss = celoss(pop,top)
		# avgloss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	# print(avgloss/3)
'''
results = []

gnames = os.listdir("test")
for gname in gnames:
	path = os.path.join("test",gname)
	frames = frameExtractor(path)
	fvect = []
	for frame in frames:
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
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