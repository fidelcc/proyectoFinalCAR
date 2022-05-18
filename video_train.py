#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse as arg
import time
from datetime import datetime
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary
from dataset import MyDataset
from modelV import LipImages


def show_lr(optimizer):
	lr = []
	for param_group in optimizer.param_groups:
		lr += [param_group['lr']]
	return np.array(lr).mean()


def ctc_decode(y):
	y = y.argmax(-1)
	return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def test(model, net, opt, kwargs, gpuAvailable):
	with torch.no_grad():
		dataset = MyDataset(opt.PATH_LIP, opt.PATH_ALING, opt.DATASET_VAL, opt.VID_PADDING, opt.TXT_PADDING, 'test')
		print('num_test_data:{}'.format(len(dataset.data)))
		model.eval()
		loader = DataLoader(dataset, batch_size=opt.BATCH_SIZE, shuffle=False, drop_last=False, **kwargs)
		loss_list = []
		wer = []
		cer = []
		crit = nn.CTCLoss()
		tic = time.time()
		for (i_iter, input) in enumerate(loader):
			if gpuAvailable:
				vid = input.get('vid').cuda()
				txt = input.get('txt').cuda()
				vid_len = input.get('vid_len').cuda()
				txt_len = input.get('txt_len').cuda()
			else:
				vid = input.get('vid')
				txt = input.get('txt')
				vid_len = input.get('vid_len')
				txt_len = input.get('txt_len')

			y = net(vid)

			loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
			loss_list.append(loss)
			pred_txt = ctc_decode(y)

			truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
			wer.extend(MyDataset.wer(pred_txt, truth_txt))
			cer.extend(MyDataset.cer(pred_txt, truth_txt))
			if (i_iter % opt.DISPLAY == 0):
				v = 1.0 * (time.time() - tic) / (i_iter + 1)
				eta = v * (len(loader) - i_iter) / 3600.0

				print(''.join(101 * '*'))
				print('{:<50} {:>50}'.format('Predicción', 'Real'))
				print(''.join(101 * '*'))
				for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
					print('{:<50}-{:>50}'.format(predict, truth))
				print(''.join(101 * '*'))
				print('test_iter={},eta={},wer={},cer={}'.format(i_iter, eta, np.array(wer).mean(), np.array(cer).mean()))
				print(''.join(101 * '*'))

		return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())


def train(model, net, opt, kwargs, gpuAvailable):
	writer = SummaryWriter()
	dataset = MyDataset(opt.PATH_LIP, opt.PATH_ALING, opt.DATASET_TRAIN, opt.VID_PADDING, opt.TXT_PADDING, 'train')
	# loader = dataset2dataloader(dataset, opt)
	loader = DataLoader(dataset, batch_size=opt.BATCH_SIZE, shuffle=True, drop_last=False, **kwargs)
	optimizer = optim.Adam(model.parameters(), lr=opt.LR, weight_decay=0., amsgrad=True)
	print('num_train_data:{}'.format(len(dataset.data)))
	crit = nn.CTCLoss()
	tic = time.time()
	train_wer = []
	for epoch in range(opt.EPOCH):
		for (i_iter, input) in enumerate(loader):
			model.train()
			if gpuAvailable:
				vid = input.get('vid').cuda()
				txt = input.get('txt').cuda()
				vid_len = input.get('vid_len').cuda()
				txt_len = input.get('txt_len').cuda()
			else:
				vid = input.get('vid')
				txt = input.get('txt')
				vid_len = input.get('vid_len')
				txt_len = input.get('txt_len')
			optimizer.zero_grad()
			y = net(vid)
			loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
			loss.backward()
			if (opt.IS_OPTIMIZE):
				optimizer.step()

			tot_iter = i_iter + epoch * len(loader)
			pred_txt = ctc_decode(y)
			truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
			train_wer.extend(MyDataset.wer(pred_txt, truth_txt))

			if (tot_iter % opt.DISPLAY == 0):
				v = 1.0 * (time.time() - tic) / (tot_iter + 1)
				eta = (len(loader) - i_iter) * v / 3600.0

				writer.add_scalar('train loss', loss, tot_iter)
				writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
				# writer.add_graph(model,  verbose=True)

				print(''.join(101 * '*'))
				print('{:<50}  {:>50}'.format('Predicción', 'Real'))
				print(''.join(101 * '*'))

				for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
					print('{:<50}-{:>50}'.format(predict, truth))
				print(''.join(101 * '*'))
				print('epoca={},itaracion={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
				print(''.join(101 * '*'))

			if (tot_iter % opt.TEST_STEP == 0):
				(loss, wer, cer) = test(model, net, opt, kwargs, device)
				print('i_iter={},lr={},loss={},wer={},cer={}'.format(tot_iter, show_lr(optimizer), loss, wer, cer))
				writer.add_scalar('val loss', loss, tot_iter)
				writer.add_scalar('wer', wer, tot_iter)
				writer.add_scalar('cer', cer, tot_iter)
				savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.SAVE_WEIGHTS, loss, wer, cer)
				(path, name) = os.path.split(savename)
				print(path, name)
				if (not os.path.exists(path)):
					os.makedirs(path)
				torch.save(model.state_dict(), savename)
				if (not opt.IS_OPTIMIZE):
					exit()


if (__name__ == '__main__'):
	print("Entrenamiento de la Red solo imágenes")
	print("Hora inicio: ", datetime.today())

	parser = arg.ArgumentParser(description='Arguments')
	# parser.add_argument('--GPU', default='0', help='GPU a usar')
	parser.add_argument('--NUM_WORKERS', default=20, type=int, help='Número de cores')
	parser.add_argument('--EPOCH', default=100, type=int, help='Número de épocas')
	parser.add_argument('--BATCH_SIZE', default=20, type=int, help='Tamaño de batch')
	parser.add_argument('--LR', default=2e-4, type=float, help='tasa de aprendizaje')
	parser.add_argument('--SEED', default=0, type=int, help='Semilla aleatoria')
	parser.add_argument('--DISPLAY', default=100, type=int, help='')
	parser.add_argument('--TEST_STEP', default=1441, type=int, help='')
	parser.add_argument('--PATH_LIP', default='/home/fidel/tesisEspecialidad/tesisFidel/datasetF/GRID/GRID_EXPERIMENT/lip', help='Path del conjunto de imagenes')
	parser.add_argument('--PATH_ALING', default='/home/fidel/tesisEspecialidad/tesisFidel/datasetF/GRID/GRID_EXPERIMENT/GRID_align_txt', help='Path de los textos a que hace refernecia cada video')
	parser.add_argument('--DATASET_TRAIN', default=f'data/unseen_train.txt', help='Lista de imágenes para entrenamiento')
	parser.add_argument('--DATASET_VAL', default=f'data/unseen_val.txt', help='Lista de imágenes para pruebas')
	parser.add_argument('--SAVE_WEIGHTS', default=f'weights/Net_unseen', help='Lugar a guardar los pesos')
	parser.add_argument('--VID_PADDING', default=75, type=int, help='')
	parser.add_argument('--TXT_PADDING', default=200, type=int, help='')
	parser.add_argument('--IS_OPTIMIZE', default=True, type=bool, help='')
	parser.add_argument('--WEIGHTS', default='', help='')
	# ./entre.sh

	opt = parser.parse_args()
	for arg in vars(opt):
		print('opt: {}={}'.format(arg, getattr(opt, arg)))

	gpuAvailable = torch.cuda.is_available()
	device = torch.device("cuda" if gpuAvailable else "cpu")
	kwargs = {"num_workers": opt.NUM_WORKERS, "pin_memory": True} if gpuAvailable else {}
	print(kwargs, device)

	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
	# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
	model = LipImages()
	if torch.cuda.device_count() > 1:
		model1 = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
	# net = nn.DataParallel(model).to(device)
	net = model.to(device)
	print(net)

	if (opt.WEIGHTS):
		pretrained_dict = torch.load(opt.WEIGHTS)
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
		missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
		print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
		print('miss matched params:{}'.format(missed_params))
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

	torch.manual_seed(opt.SEED)
	if gpuAvailable:
		torch.cuda.manual_seed_all(opt.SEED)
		summary(model, input_size=(3, 75, 64, 128))
	else:
		summary(model, input_size=(3, 75, 64, 128), device=device)

	train(model, net, opt, kwargs, gpuAvailable)
	print("Hora término: ", datetime.today())
	exit(0)
