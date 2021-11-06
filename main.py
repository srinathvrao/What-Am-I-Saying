import cv2
import numpy as np
import os
import handshape_feature_extractor 
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

middle_frames = []
extractor = HandShapeFeatureExtractor()

middle_frames = []
outputs = []
gnames = sorted(os.listdir("traindata/"))
c=0
for gname in gnames:
	path = os.path.join("traindata",gname)
	frame = frameExtractor(path)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fvect = extractor.extract_feature(frame)[0]
	middle_frames.append(np.array(fvect))
	outputs.append(c)
	c+=1

results = []

gnames = os.listdir("test")
for gname in gnames:
	path = os.path.join("test",gname)
	frame = frameExtractor(path)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	fvect = extractor.extract_feature(frame)[0]
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
	print(gname, outputs[mindisti])
	# results.append([ svc.predict([fvect])[0] ])
filename = "results.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(results)