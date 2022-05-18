import glob
import os
from itertools import groupby
import shutil

PATH = os.path.abspath(os.getcwd())
PATH_WRITE = PATH + "GRID_EXPERIMENT/lip/"
addPath="video/mpg_6000"
PATH_READ_WAV = PATH + "GRID_ORIGINAL/*/*/"
print(PATH_READ_WAV)

READ_WAV =  glob.glob(PATH_READ_WAV+"*.wav")
skipDir = PATH_READ_WAV.count('/') #+ 1
print(skipDir)

wav_filenames = map(lambda filename: (filename.split("/")[skipDir], filename), READ_WAV)
wav_filenames = sorted(wav_filenames)

TodosWav = 0
for name_wav, Dir_wav in wav_filenames:
	#print(name_wav)
	sourcePath = Dir_wav
	nameDir = name_wav.split(".")[0]
	nameDir = nameDir + "/" + name_wav
	
	replaceString = Dir_wav.split("/")[9]
	Dir_wav = Dir_wav.replace(replaceString, "video/mpg_6000")
	Dir_wav = Dir_wav.replace("GRID_ORIGINAL", "GRID_EXPERIMENT/lip")
	Dir_wav = Dir_wav.replace(name_wav, nameDir)
	try:
		#shutil.copyfile(sourcePath, Dir_wav)
		print(sourcePath, Dir_wav)
	except:
		print("Problema: ", sourcePath)
	
	TodosWav=TodosWav+1
	if (TodosWav%1000==0):
		#print(sourcePath, Dir_wav)
		print(TodosWav)
print(TodosWav)

