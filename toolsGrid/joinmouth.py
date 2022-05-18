import cv2
import numpy as np
import glob
import os
from itertools import groupby
from matplotlib import pyplot as plt
#from PIL import Image
from tqdm.notebook import tqdm

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face
import time
import torch

PATH_DIR = os.path.abspath(os.getcwd())
PATH_WRITE = PATH_DIR + "/datasetF/GRID/GRID_EXPERIMENT/lip/*/*/*/"
print(PATH_WRITE)
images_filenames = glob.glob(PATH_WRITE + "*") #mpg, mp3
for dirFiles in images_filenames:
	imagenes = glob.glob(dirFiles + "/*.jpg")
	skipDir = dirFiles.count('/')
	nameImg= dirFiles.split("/")[skipDir]
	roiFile = dirFiles + "/" + nameImg + ".png"
	roiSequence = list()
	for imgs in imagenes:
		grayed = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)
		grayed = grayed/255
		grayed = cv2.resize(grayed, (224,224))
		roiSequence.append(grayed)

	cv2.imwrite(roiFile, np.floor(255*np.concatenate(roiSequence, axis=1)).astype(np.int))

	inp = np.stack(roiSequence, axis=0)
	inp = np.expand_dims(inp, axis=[1,2])
	inp = (inp - normMean)/normStd
	inputBatch = torch.from_numpy(inp)
	inputBatch = (inputBatch.float()).to(device)
	vf.eval()
	with torch.no_grad():
		outputBatch = vf(inputBatch)
	out = torch.squeeze(outputBatch, dim=1)
	out = out.cpu().numpy()
	np.save(visualFeaturesFile, out)

exit(0)
	
	
	
