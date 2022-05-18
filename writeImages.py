import cv2
import numpy as np
import glob
import os
from itertools import groupby
#from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, extract_face

PATH_DIR = os.path.abspath(os.getcwd())
PATH_WRITE = PATH_DIR + "/datasetF/GRID/GRID_IMAGES/"
detector = MTCNN(select_largest=False, post_process=False, device='cuda')
print(detector)
print(PATH_WRITE)
video_filenames = PATH_DIR + "/datasetF/GRID/GRID_ORIGINAL"
skipDir = video_filenames.count('/') + 1
print(skipDir)

#face_cascade = cv2.CascadeClassifier(PATH_DIR + '/haarcascade_frontalface_default.xml')
video_filenames = glob.glob(video_filenames + "/*/*/*.mpg") #mpg, mp3
personas=[]

video_filenames = map(lambda filename: (filename.split("/")[skipDir], filename), video_filenames)
video_filenames = sorted(video_filenames)

TodosVideos = 0
contador = 0
for personasDir, personas_videos in groupby(video_filenames, lambda x: x[0]):
	listVidxPersona = list(personas_videos)
	totalVideos = len(listVidxPersona)
	personas.append(personasDir)
	TodosVideos = TodosVideos + totalVideos
	contador = contador + 1
	for i, videoPersona in enumerate(listVidxPersona):
		video = videoPersona[1].split("/")
		directory = video[-1].split(".")
		path = os.path.join(PATH_WRITE + personasDir, "video/mpg_6000", directory[0])
		try:
			os.makedirs(path, exist_ok = True)
		except OSError as error:
			print("El directorio '%s' no se pudo crear" % directory[0])
		
		cap = cv2.VideoCapture(videoPersona[1])
		if (cap.isOpened()== False): 
			print("Error al abrir el video ", videoPersona[1])
		v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frames = []
		for i in range(v_len):
			success = cap.grab()
			success, frame = cap.retrieve()
			if not success:
				continue
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frames.append(Image.fromarray(frame))
		faces = detector(frames)
		countFrame = 1
		
		for frame_faces in faces:
			try:
				c, w, h = frame_faces.shape
				frame_faces = frame_faces.permute(1, 2, 0).int().numpy()[int(h/2):h, 0:w ]
				plt.figure(figsize=(2.07, 1.4), dpi=100)
				plt.axis('off')
				plt.imshow(frame_faces)
				plt.savefig(path + '/%d.png' % countFrame, bbox_inches='tight', pad_inches = 0)
				plt.close()
				plt.cla()
				plt.clf()
				countFrame = countFrame+1
			except OSError as error:
				print("Error al generar las imágenes en ", path)

print(contador, TodosVideos, personas)

