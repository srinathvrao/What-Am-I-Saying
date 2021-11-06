import cv2
import glob
import os
import numpy as np
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from scipy import spatial
import csv
from csv import reader

extractor = HandShapeFeatureExtractor().get_instance()

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================

trainingDataCSV = "trainingData.csv"

'''

train_middle_penult = []
gnames = sorted(glob.glob("traindata/*.mp4"))
c=0
for path in gnames:
	print(path)
	frameExtractor(path,"trainframes",c)
	img = cv2.imread("trainframes" + "/%#05d.png" % (c+1))
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fvect = np.squeeze(extractor.extract_feature(frame))
	train_middle_penult.append(fvect)
	c+=1

np.savetxt(trainingDataCSV,train_middle_penult,delimiter=",")


'''

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================

test_middle_penult = []
gnames = glob.glob("test/*.mp4")
testingDataCSV = "testingData.csv"
c=0
for path in gnames:
	frameExtractor(path,"testframes",c)
	img = cv2.imread("testframes" + "/%#05d.png" % (c+1))
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	fvect = np.squeeze(extractor.extract_feature(frame).squeeze())
	test_middle_penult.append(fvect)

np.savetxt(testingDataCSV,test_middle_penult,delimiter=",")

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

trainingData = np.genfromtxt(trainingDataCSV,delimiter=",")
testData = np.genfromtxt(testingDataCSV,delimiter=",")

if len(testData.shape)==1:
	testData = np.array([testData])

result = []
indices = []

for test in testData:
	cossimlist = []
	for trv in trainingData:
		cossimlist.append(spatial.distance.cosine(test, trv))
	print(cossimlist.index(min(cossimlist)))
	result.append(int(cossimlist.index(min(cossimlist))))

filename = "Results.csv"
np.savetxt(filename,result,delimiter=",",fmt="% d")
# with open(filename, 'w') as csvfile: 
# 	csvwriter = csv.writer(csvfile)
# 	csvwriter.writerows(results)

