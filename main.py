import cv2
import os
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

middle_frames = []
extractor = HandShapeFeatureExtractor().get_instance()

gnames = sorted(os.listdir("traindata/"))
c=0
for gname in gnames:
	# print(gname)
	path = os.path.join("traindata",gname)
	frameExtractor(path,"trainframes",c)
	img = cv2.imread("trainframes" + "/%#05d.png" % (c+1))
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	fvect = extractor.extract_feature(frame).squeeze()
	middle_frames.append(fvect)
	c+=1
results = []

gnames = os.listdir("test/")
c=0
for gname in gnames:
	path = os.path.join("test",gname)
	frameExtractor(path,"testframes",c)
	img = cv2.imread("testframes" + "/%#05d.png" % (c+1))
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
	fvect = extractor.extract_feature(frame).squeeze()
	cc=0
	mindisti, mindist = 0,1e8
	for v in middle_frames:
		cosdist = spatial.distance.cosine(fvect,v)
		if cosdist<mindist:
			mindist = cosdist
			mindisti = cc
		cc+=1
	print(gname, mindisti)
	results.append([ mindisti ])
	c+=1

filename = "results.csv"
with open(filename, 'w') as csvfile: 
	csvwriter = csv.writer(csvfile)
	csvwriter.writerows(results)