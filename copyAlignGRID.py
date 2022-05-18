import glob
import os
from itertools import groupby
import shutil

PATH = os.path.abspath(os.getcwd())
PATH_WRITE = PATH + "GRID_EXPERIMENT/lip/"
PATH_READ_ALIGN = PATH + "GRID_ORIGINAL/*/*/"
print(PATH_READ_ALIGN)

READ_ALIGN =  glob.glob(PATH_READ_ALIGN+"*.align")
skipDir = PATH_READ_ALIGN.count('/') #+ 1
print(skipDir)

align_filenames = map(lambda filename: (filename.split("/")[skipDir], filename), READ_ALIGN)
align_filenames = sorted(align_filenames)
print(align_filenames)
TodosAlign = 0
for name_align, Dir_align in align_filenames:
	#print(name_align)
	sourcePath = Dir_align
	nameDir = name_align.split(".")[0]
	nameDir = nameDir + "/" + name_align
	
	replaceString = Dir_align.split("/")[9]
	Dir_align = Dir_align.replace(replaceString, "video/mpg_6000")
	Dir_align = Dir_align.replace("GRID_ORIGINAL", "GRID_EXPERIMENT/lip")
	Dir_align = Dir_align.replace(name_align, nameDir)
	try:
		shutil.copyfile(sourcePath, Dir_align)
		#print(sourcePath, Dir_align)
	except:
		print("Problema: ", sourcePath)
	
	TodosAlign=TodosAlign+1
	if (TodosAlign%1000==0):
		#print(sourcePath, Dir_align)
		print(TodosAlign)

print(TodosAlign)

