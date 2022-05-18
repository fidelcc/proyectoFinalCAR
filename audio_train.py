
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import argparse as arg

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

#from models.audio_net import AudioNet
from modelA import AudioNet

from data.GRID_dataset import GRIDMain
from data.utils import collate_fn
from utils.general import num_params, train, evaluate


def trainModel(model, net, opt, kwargs, gpuAvailable):


	audioParams = {"stftWindow": opt.STFT_WINDOW, "stftWinLen": opt.STFT_WIN_LENGTH, "stftOverlap": opt.STFT_OVERLAP}
	noiseParams = {"noiseFile": opt.PATH_DATA + "/noise.wav", "noiseProb": opt.NOISE_PROBABILITY, "noiseSNR": opt.NOISE_SNR_DB}
	trainData = GRIDMain("train", opt.PATH_DATA, opt.MAIN_REQ_INPUT_LENGTH, opt.CHAR_TO_INDEX, opt.STEP_SIZE, audioParams, noiseParams)
	print(trainData)
	trainLoader = DataLoader(trainData, batch_size=opt.BATCH_SIZE, collate_fn=collate_fn, shuffle=True, **kwargs)
	noiseParams = {"noiseFile": opt.PATH_DATA + "/noise.wav", "noiseProb": 0, "noiseSNR": opt.NOISE_SNR_DB}
	valData = GRIDMain("val", opt.PATH_DATA, opt.MAIN_REQ_INPUT_LENGTH, opt.CHAR_TO_INDEX, opt.STEP_SIZE, audioParams, noiseParams)
	valLoader = DataLoader(valData, batch_size = opt.BATCH_SIZE, collate_fn=collate_fn, shuffle=True, **kwargs)

	optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.MOMENTUM1, opt.MOMENTUM2))
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=opt.LR_SCHEDULER_FACTOR, patience=opt.LR_SCHEDULER_WAIT, threshold=opt.LR_SCHEDULER_THRESH,
													 threshold_mode="abs", min_lr=opt.FINAL_LR, verbose=True)
	loss_function = nn.CTCLoss(blank=0, zero_infinity=False)

	trainingLossCurve = list()
	validationLossCurve = list()
	trainingWERCurve = list()
	validationWERCurve = list()

	numTotalParams, numTrainableParams = num_params(model)
	print("\nNúmero de parámetros en el modelo = %d" % (numTotalParams))
	print("Número de parámetros entrenables = %d\n" % (numTrainableParams))

	trainParams = {"spaceIx": opt.CHAR_TO_INDEX[" "], "eosIx": opt.CHAR_TO_INDEX["<EOS>"]}
	valParams = {"decodeScheme": "greedy", "spaceIx": opt.CHAR_TO_INDEX[" "], "eosIx": opt.CHAR_TO_INDEX["<EOS>"]}
	
	print("\nEntrenando .... \n")
	for step in range(opt.EPOCH):
		trainingLoss, trainingCER, trainingWER = train(model, trainLoader, optimizer, loss_function, device, trainParams, opt.INDEX_TO_CHAR)
		trainingLossCurve.append(trainingLoss)
		trainingWERCurve.append(trainingWER)

		validationLoss, validationCER, validationWER = evaluate(model, valLoader, loss_function, device, valParams, opt.INDEX_TO_CHAR)
		validationLossCurve.append(validationLoss)
		validationWERCurve.append(validationWER)

		print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
			  % (step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))

		scheduler.step(validationWER)

		if ((step % opt.SAVE_FREQUENCY == 0) or (step == opt.EPOCH - 1)) and (step != 0):
			savePath = opt.CODE_DIRECTORY + "/checkpoints/models/train-step_{:04d}-wer_{:.3f}.pt".format(step, validationWER)
			torch.save(model.state_dict(), savePath)

			plt.figure()
			plt.title("Loss Curves")
			plt.xlabel("Step No.")
			plt.ylabel("Loss value")
			plt.plot(list(range(1, len(trainingLossCurve) + 1)), trainingLossCurve, "blue", label="Train")
			plt.plot(list(range(1, len(validationLossCurve) + 1)), validationLossCurve, "red", label="Validation")
			plt.legend()
			plt.savefig(opt.CODE_DIRECTORY + "/checkpoints/plots/train-step_{:04d}-loss.png".format(step))
			plt.close()

			plt.figure()
			plt.title("WER Curves")
			plt.xlabel("Step No.")
			plt.ylabel("WER")
			plt.plot(list(range(1, len(trainingWERCurve) + 1)), trainingWERCurve, "blue", label="Train")
			plt.plot(list(range(1, len(validationWERCurve) + 1)), validationWERCurve, "red", label="Validation")
			plt.legend()
			plt.savefig(opt.CODE_DIRECTORY + "/checkpoints/plots/train-step_{:04d}-wer.png".format(step))
			plt.close()
	return

if __name__ == "__main__":
	#main()
	PATH_DIR = os.path.abspath(os.getcwd())
	print(PATH_DIR)
	print("Entrenamiento de la Red solo con audio")
	print("Hora inicio: ", datetime.today())
	CHAR_TO_INDEX = {" ":1, "'":22, "1":30, "0":29, "3":37, "2":32, "5":34, "4":38, "7":36, "6":35, "9":31, "8":33,
			"A":5, "C":17, "B":20, "E":2, "D":12, "G":16, "F":19, "I":6, "H":9, "K":24, "J":25, "M":18,
			"L":11, "O":4, "N":7, "Q":27, "P":21, "S":8, "R":10, "U":13, "T":3, "W":15, "V":23, "Y":14,
			"X":26, "Z":28, "<EOS>":39}
	INDEX_TO_CHAR = {1:" ", 22:"'", 30:"1", 29:"0", 37:"3", 32:"2", 34:"5", 38:"4", 36:"7", 35:"6", 31:"9", 33:"8",
                         5:"A", 17:"C", 20:"B", 2:"E", 12:"D", 16:"G", 19:"F", 6:"I", 9:"H", 24:"K", 25:"J", 18:"M",
                         11:"L", 4:"O", 7:"N", 27:"Q", 21:"P", 8:"S", 10:"R", 13:"U", 3:"T", 15:"W", 23:"V", 14:"Y",
                         26:"X", 28:"Z", 39:"<EOS>"}  

	parser = arg.ArgumentParser(description='Arguments')
	parser.add_argument('--CODE_DIRECTORY', default=PATH_DIR, help='Ruta donde se encuentra el código')
	parser.add_argument('--PATH_DATA', default="/home/fidel/tesisEspecialidad/tesisFidel/datasetF/GRID/GRID_EXPERIMENT", help='Ruta donde se encuentra el conjunto de datos')
	parser.add_argument('--MAIN_REQ_INPUT_LENGTH', default=145, type=int, help='mínimo tamaño de entrada')
	parser.add_argument('--CHAR_TO_INDEX', default=CHAR_TO_INDEX, help='caracteres')
	parser.add_argument('--INDEX_TO_CHAR', default=INDEX_TO_CHAR, help='caracteres')
	
	parser.add_argument('--AUDIO_FEATURE_SIZE', default=501, type=int, help='tamaño de las características del audio')
	parser.add_argument('--NUM_CLASSES', default=40, type=int, help='número de caracteres')
	parser.add_argument('--STFT_WINDOW', default="hamming", help='tipo de ventaneo para el análisis con STFT')
	parser.add_argument('--STFT_WIN_LENGTH', default=0.040, type=float, help='tamaño de ventana en segundos mientras se calcula STFT')
	parser.add_argument('--STFT_OVERLAP', default=0.030, type=float, help='traslape de ventanas consecutivas en segundos mientras se calcula STFT')
	parser.add_argument('--NOISE_PROBABILITY', default=0.05, type=float, help='probabilidad de ruido')
	parser.add_argument('--NOISE_SNR_DB', default=0, type=int, help='nivel de ruido')
	parser.add_argument('--PE_MAX_LENGTH', default=2500, type=int, help='longitud para calcular los encoders')
	parser.add_argument('--TX_NUM_FEATURES', default=512, type=int, help='tamaño de características de entrada a transformer')
	parser.add_argument('--TX_ATTENTION_HEADS', default=8, type=int, help='tattentions heads')
	parser.add_argument('--TX_NUM_LAYERS', default=6, type=int, help='Número de bloques transformer')
	parser.add_argument('--TX_FEEDFORWARD_DIM', default=2048, type=int, help='tamaño de la capa oculta en transformer')
	parser.add_argument('--TX_DROPOUT', default=0.1, type=float, help='dropout en transformer')
	parser.add_argument('--NUM_WORKERS', default=20, type=int, help='Número de cores')
	parser.add_argument('--EPOCH', default=50, type=int, help='Número de épocas')
	parser.add_argument('--DISPLAY', default=10, type=int, help='iteraciones para mostrar información')
	parser.add_argument('--STEP_SIZE', default=16384, type=int, help='número de ejemplos en un paso (época)')
	parser.add_argument('--BATCH_SIZE', default=20, type=int, help='Tamaño de batch')
	parser.add_argument('--LR', default=1e-4, type=float, help='tasa de aprendizaje')
	parser.add_argument('--FINAL_LR', default=1e-6, type=float, help='tasa de aprendijae final')
	parser.add_argument('--LR_SCHEDULER_FACTOR', default=0.5, type=float, help='factor de la tasa de aprendizaje')
	parser.add_argument('--LR_SCHEDULER_WAIT', default=15, type=int, help='número de pasos de espera para el cambio de tasa de aprendizaje')
	parser.add_argument('--LR_SCHEDULER_THRESH', default=0.001, type=float, help='threshold para wer')
	parser.add_argument('--MOMENTUM1', default=0.9, type=float, help='momento 1')
	parser.add_argument('--MOMENTUM2', default=0.999, type=float, help='momento 2')
	parser.add_argument('--SEED', default=0, type=int, help='Semilla aleatoria')
	parser.add_argument('--SAVE_FREQUENCY', default=10000, type=int, help='')
	parser.add_argument('--PATH_LIP', default='/home/fidel/tesisEspecialidad/tesisFidel/datasetF/GRID/GRID_EXPERIMENT/lip', help='Path del conjunto de imagenes')
	parser.add_argument('--PATH_ALING', default='/home/fidel/tesisEspecialidad/tesisFidel/datasetF/GRID/GRID_EXPERIMENT/GRID_align_txt', help='Path de los textos a que hace refernecia cada video')
	parser.add_argument('--DATASET_TRAIN', default=f'data/unseen_train.txt', help='Lista de imágenes para entrenamiento')
	parser.add_argument('--DATASET_VAL', default=f'data/unseen_val.txt', help='Lista de imágenes para pruebas')
	parser.add_argument('--SAVE_WEIGHTS', default=f'weights/Net_unseen', help='Lugar a guardar los pesos')
	parser.add_argument('--VID_PADDING', default=75, type=int, help='')
	parser.add_argument('--TXT_PADDING', default=200, type=int, help='')
	parser.add_argument('--IS_OPTIMIZE', default=True, type=bool, help='')
	parser.add_argument('--WEIGHTS', default='', help='')

	opt = parser.parse_args()
	for arg in vars(opt):
		print('opt: {}={}'.format(arg, getattr(opt, arg)))
	

	gpuAvailable = torch.cuda.is_available()
	device = torch.device("cuda" if gpuAvailable else "cpu")
	kwargs = {"num_workers": opt.NUM_WORKERS, "pin_memory": True} if gpuAvailable else {}
	print(kwargs,device)
	
	
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
	
	#model = LipImages()
	model = AudioNet(opt.TX_NUM_FEATURES, opt.TX_ATTENTION_HEADS, opt.TX_NUM_LAYERS, opt.PE_MAX_LENGTH, opt.AUDIO_FEATURE_SIZE, opt.TX_FEEDFORWARD_DIM, opt.TX_DROPOUT, opt.NUM_CLASSES)
	
	#model.to(device)
	#print(model)
	
	
	if torch.cuda.device_count() > 1:
		model1 = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
	# net = nn.DataParallel(model).to(device)
	net = model.to(device)
	print(net)



	if os.path.exists(opt.CODE_DIRECTORY + "/checkpoints"):
		while True:
			ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
			if ch == "y":
				break
			elif ch == "n":
				exit()
			else:
				print("Invalid input")
		shutil.rmtree(opt.CODE_DIRECTORY + "/checkpoints")

	os.mkdir(opt.CODE_DIRECTORY + "/checkpoints")
	os.mkdir(opt.CODE_DIRECTORY + "/checkpoints/models")
	os.mkdir(opt.CODE_DIRECTORY + "/checkpoints/plots")

	if (opt.WEIGHTS):
		print("\n\nPre-trained Model File: %s" % (opt.WEIGHTS))
		print("\nLoading the pre-trained model .... \n")
		model.load_state_dict(torch.load(opt.CODE_DIRECTORY + opt.WEIGHTS, map_location=device))
		#model.to(device)
		print("Loading Done.\n")

	#if (opt.WEIGHTS):
	#	pretrained_dict = torch.load(opt.WEIGHTS)
	#	model_dict = model.state_dict()
	#	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
	#	missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
	#	print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
	#	print('miss matched params:{}'.format(missed_params))
	#	model_dict.update(pretrained_dict)
	#	model.load_state_dict(model_dict)
	#matplotlib.use(opt.SEED)
	np.random.seed(opt.SEED)
	torch.manual_seed(opt.SEED)

	
	if gpuAvailable:
		torch.cuda.manual_seed_all(opt.SEED)
		#summary(model, input_size=(3, 75, 64, 128))
		#else:
		#summary(model, input_size=(3, 75, 64, 128), device=device)

	trainModel(model, net, opt, kwargs, gpuAvailable)
	print("Hora término: ", datetime.today())
	exit(0)
	
	
	
	
	
	
	
	
