import argparse
import os
import random
import shutil
import time
import warnings
import builtins
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import groupNorm.resnet as models_GN
from torch import randperm

import numpy as np
import sys
from tensorboardX import SummaryWriter
from yfcc100m_dataset import YFCC_CL_Dataset_offline_val, YFCC_CL_Dataset_online
import psutil
import gc
import copy


model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
					choices=model_names,
					help='model architecture: ' +
						' | '.join(model_names) +
						' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('-wf', '--write-freq', default=100, type=int,
					metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')
# used for continual learning
parser.add_argument('--cell_id', default='../final_metadata/cellID_yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250.npy', type=str,
					help='file that store the cell IDS')

parser.add_argument('--use_val', default=0, 
					type=int, help='whether to use validation set')
parser.add_argument('--root', default="/export/share/Datasets/yfcc100m_full_dataset/images/", 
					type=str, help='root to the image dataset, used to save the memory consumption')
parser.add_argument('--data', default="/export/share/Datasets/yfcc100m_full_dataset/metadata_geolocation/", 
					type=str, help='path to the training data')
parser.add_argument('--data_val', default="/export/share/Datasets/yfcc100m_full_dataset/metadata_geolocation/yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250_valid_files_2004To2014_compact_val.csv", 
					type=str, help='path to the metadata')

parser.add_argument('--gradient_steps_per_batch', default=1, 
					type=int, help='number of gradient steps per batch, used to compare against firehose paper')
parser.add_argument('--size_replay_buffer', default=0, 
					type=int, help='initial size of the experience replay buffer (per gpu, so if you have 8 gpus, each gpu will have size_replay_buffer number of samples in the buffer)')
parser.add_argument('--sample_rate_from_buffer', default=0.5, 
					type=float, help='how many samples are from the buffer')
parser.add_argument('--sampling_strategy', default='FIFO', 
					type=str, help='sampling strategy (currently only support FIFO)')
parser.add_argument('--num_classes', default=500, 
					type=int, help='number of classes (used only for constructing ring buffer, no need to set this value)')

# separate the hyperparameters (lr, bs, data aug, bn statistics, and so on) on old and new data
parser.add_argument('--weight_old_data', default=1.0, 
					type=float, help='weight of the loss on old data, used to separate the effect of learning rate on old and new data')

parser.add_argument('--NOSubBatch', default=1, 
					type=int, help='separate each batch into consecutive sub-batches, used only in online mode, for testing the effect of batch size')


parser.add_argument('--GN', default=0, 
					type=int, help='number of channels per group. If it is 0, it means '
					'batch norm instead of group-norm')

parser.add_argument('--used_data_start', default = 0, type = int, help = 'number of images used, use the full dataset if set to > 1')
parser.add_argument('--used_data_end', default = -1, type = int, help = 'number of images used, use the full dataset if set to > 1')

# separate between pure-replay or mixed replay
parser.add_argument("--ReplayType", default = 'mixRep', 
					choices=['mixRep'], help='Type of replay buffer')
parser.add_argument("--SaveInter", default = 1, type = int, help = 'whether to save intermediate models')

# param for population-based adaptive LR
parser.add_argument("--NOModels", default = 3, type = int, help = 'how many models to train simultaneously')
parser.add_argument("--LRMultiFactor", default = 2.0, type = float, help = 'factor to change LR between models')
parser.add_argument("--LR_adjust_intv", default = 5, type = int, help = 'adjust LR in how many epochs')

parser.add_argument("--ABS_performanceGap", default = 0.5, type = float, help = 'the allowed performance gap between the accuracy of old and new samples, hyper-param for adaptive buffer size (tuned through cross val)')
parser.add_argument("--use_ADRep", default=1, type = int, help = 'whether to use ADRep')
best_acc1 = 0



def main():
	args = parser.parse_args()

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	global best_acc1

	args.gpu = gpu

	# suppress printing if not master
	if args.multiprocessing_distributed and args.gpu != 0:
		def print_pass(*args):
			pass
		builtins.print = print_pass

	args.output_dir = create_output_dir(args)
	os.makedirs(args.output_dir, exist_ok = True)
	writer = SummaryWriter(args.output_dir)

	print("output_dir = {}".format(args.output_dir))    

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	
	model, criterion, optimizer, num_classes = init_model(args, ngpus_per_node)

	args.num_classes = num_classes

	# init the full dataset
	train_set, val_set = init_dataset(args)
	cudnn.benchmark = True

	if args.evaluate:
		# optionally resume from a checkpoint
		if args.resume:
			if os.path.isfile(args.resume):
				args.start_epoch, _, _, _ = resume_online_eval(args, model, optimizer) 
			else:
				raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

		out_folder_eval = args.output_dir + '/epoch{}'.format(args.start_epoch)
		os.makedirs(out_folder_eval, exist_ok = True)
		writer = SummaryWriter(out_folder_eval)
		print("[validation]: saving validation result to {}".format(out_folder_eval))
		val_loader = init_val_loader(args, val_set)
		acc1 = validate(val_loader, model, criterion, args, writer, args.epochs, is_forward = 2)

		# do forward and backward transfer too
		# 1. read out time of current example
		time_taken = torch.load(args.data+'train_time.torchSave')
		idx_last = compute_idx_curr(len(time_taken), args.start_epoch, args,  ngpus_per_node)
		if idx_last < 0:
			time_last = time_taken[0] - 1
		else:
			time_last = time_taken[min(idx_last, len(time_taken)-1)]

		# compute forward 
		val_set.set_transfer_time_point(args, val_set, time_last, is_forward = True)
		val_loader = init_val_loader(args, val_set)
		out_folder_eval = args.output_dir + '/epoch{}/forward'.format(args.start_epoch)
		os.makedirs(out_folder_eval, exist_ok = True)
		writer = SummaryWriter(out_folder_eval)
		print("[forward transfer]: saving validation result to {}".format(out_folder_eval))
		acc1 = validate(val_loader, model, criterion, args, writer, args.epochs, is_forward = 1, time_last = time_last)
		# backward transfer
		val_set.set_transfer_time_point(args, val_set, time_last, is_forward = False)
		val_loader = init_val_loader(args, val_set)
		out_folder_eval = args.output_dir + '/epoch{}/backward'.format(args.start_epoch)
		os.makedirs(out_folder_eval, exist_ok = True)
		writer = SummaryWriter(out_folder_eval)
		print("[backward transfer]: saving validation result to {}".format(out_folder_eval))
		acc1 = validate(val_loader, model, criterion, args, writer, args.epochs, is_forward = 0, time_last = time_last)

	else:
		train_online(args, model, criterion, optimizer, train_set, val_set, ngpus_per_node, writer)

def compute_idx_curr(data_size, epoch, args, ngpus_per_node):
	data_size_per_epoch = math.ceil(data_size/args.epochs)
	if data_size_per_epoch % (args.batch_size* ngpus_per_node) != 0:
		data_size_per_epoch = data_size_per_epoch - data_size_per_epoch % (args.batch_size* ngpus_per_node) + (args.batch_size* ngpus_per_node)    
	return data_size_per_epoch * epoch - 1 


def init_model(args, ngpus_per_node):
	cell_ids = np.load(args.cell_id, allow_pickle = True)
	num_classes = cell_ids.size + 1 

	print("init with num_classes = {}".format(num_classes))

	if args.GN == 0:
		 # create model
		if args.pretrained:
			print("=> using pre-trained model '{}'".format(args.arch))
			model = models.__dict__[args.arch](pretrained=True)
		else:
			print("=> creating model '{}'".format(args.arch))
			model = models.__dict__[args.arch](num_classes = num_classes, norm_layer = nn.SyncBatchNorm)
	else:
		 # create model
		if args.pretrained:
			print("=> using pre-trained model '{}'".format(args.arch))
			model = models_GN.__dict__[args.arch](pretrained=True,
										group_norm=args.GN)
		else:
			print("=> creating model '{}'".format(args.arch))
			model = models_GN.__dict__[args.arch](num_classes = num_classes,
												group_norm=args.GN)

	if not torch.cuda.is_available():
		print('using CPU, this will be slow')
	elif args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
									momentum=args.momentum,
									weight_decay=args.weight_decay)
	
	return model, criterion, optimizer, num_classes


def init_dataset(args):
	# Data loading code
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	
	trans = transforms.Compose([            
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])
	
	trans_test = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])
	
	print("data augmentation = {}; data augmentation for test batch = {}".format(trans, trans_test))
	
	if args.evaluate:
		train_dataset = None
		val_dataset = YFCC_CL_Dataset_offline_val(args,
		transform = trans_test)
	else:		
		train_dataset = YFCC_CL_Dataset_online(args,
			transform = trans, transform_RepBuf = trans, trans_test = trans_test)
		val_dataset = YFCC_CL_Dataset_offline_val(args,
			transform = trans_test)

	return train_dataset, val_dataset

def resume_online(args, model_pool, optim_pool, meter_pool):
	print("=> loading checkpoint '{}'".format(args.resume))
	if args.gpu is None:
		checkpoint = torch.load(args.resume)
	else:
		# Map model to be loaded to specified single gpu.
		loc = 'cuda:{}'.format(args.gpu)
		checkpoint = torch.load(args.resume, map_location=loc)
	
	args.start_epoch = checkpoint['epoch']
	best_acc1 = checkpoint['best_acc1']
	
	for i in range(len(model_pool)):
		model_pool[i].load_state_dict(checkpoint['state_dict{}'.format(i)])
		optim_pool[i].load_state_dict(checkpoint['optimizer{}'.format(i)])
		meter_pool[i] = checkpoint['online_fit_meters{}'.format(i)]
	
	userID_last = checkpoint['userID_last']

	print("=> loaded checkpoint '{}' (epoch {})"
		  .format(args.resume, checkpoint['epoch']))

	args.size_replay_buffer = checkpoint['size_replay_buffer']
	return checkpoint['epoch'], checkpoint['best_acc1'], checkpoint['online_fit_meters'], userID_last 

def resume_online_eval(args, model, optimizer):
	print("=> loading checkpoint '{}'".format(args.resume))

	if args.gpu is None:
		checkpoint = torch.load(args.resume)
	else:
		# Map model to be loaded to specified single gpu.
		# zhipeng remember to change here to save GPU memory!!!!!!!!!!!!
		loc = 'cuda:{}'.format(args.gpu)
		checkpoint = torch.load(args.resume, map_location=loc)
	
	args.start_epoch = checkpoint['epoch']
	best_acc1 = checkpoint['best_acc1']
	
	model.load_state_dict(checkpoint['state_dict'])
	
	optimizer.load_state_dict(checkpoint['optimizer'])

	online_fit_meters = checkpoint['online_fit_meters']
	
	userID_last = checkpoint['userID_last']

	print("=> loaded checkpoint '{}' (epoch {})"
		  .format(args.resume, checkpoint['epoch']))

	return checkpoint['epoch'], checkpoint['best_acc1'], online_fit_meters, userID_last 


def init_val_loader(args, val_set):
	return torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

def init_loader_v2(args, train_set, epoch):
	train_set._change_data_range(epoch = epoch)
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = False)
	else:
		train_sampler = None

	print("sampler.shuffle = {}".format(train_sampler.shuffle))
	
	batchSize4Loader = int(args.batch_size/args.NOSubBatch)
		
	print("[initLoaderv2]: batchSize4Loader = {}".format(batchSize4Loader))
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=batchSize4Loader, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)

	print("train_loader.len = {}".format(train_loader.__len__()))

	return train_loader, train_sampler


def init_online_fit_meters():
	return [AverageMeter('LossOF', ':.4e'), AverageMeter('AccOF@1', ':6.2f'), AverageMeter('AccOF@5', ':6.2f')]

def init_meters_for_hypTune():
	return [AverageMeter('Loss', ':.4e'), AverageMeter('AccF@1', ':6.2f'), AverageMeter('AccF@5', ':6.2f')]

def train_online(args, model, criterion, optimizer, train_set, val_set, ngpus_per_node, writer):
	# initialize model pool and optimizer pool for training
	model_pool = [None] * args.NOModels
	optim_pool = [None] * args.NOModels
	LR_min = args.lr * args.LRMultiFactor
	meter_pool = [None] * args.NOModels
	writer_pool = [None] * args.NOModels

	for i in range(0, args.NOModels):
		if args.NOModels == 1 or i == 1:
			model_pool[i] = model
			optim_pool[i] = optimizer
		else:
			model_pool[i] = copy.deepcopy(model)
			optim_pool[i] = torch.optim.SGD(model_pool[i].parameters(), LR_min/float(args.LRMultiFactor**i),
									momentum=args.momentum,
									weight_decay=args.weight_decay)
		# init metrics for online fit
		meter_pool[i] = init_meters_for_hypTune()
		out_dir_curr = args.output_dir + "/model_{}".format(i)
		os.makedirs(out_dir_curr, exist_ok = True)
		writer_pool[i] = SummaryWriter(out_dir_curr)

		print("[model pool init] LR[{}] = {}".format(i, get_lr(optim_pool[i])))

	online_fit_meters = init_online_fit_meters()

	userID_last = -1

	if args.resume:
		if os.path.isfile(args.resume):
			args.start_epoch, _, online_fit_meters, userID_last = resume_online(args, model_pool, optim_pool, meter_pool)
		else:
			raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

	print("args.workers = {}".format(args.workers))	
	for epoch in range(args.start_epoch, args.epochs):
		train_loader, train_sampler = init_loader_v2(args, train_set, epoch)

		if args.distributed:
			train_sampler.set_epoch(0)
		user_ID_last, idx_best_model = train_MultiGD(train_loader, model_pool, criterion, optim_pool, epoch, args, writer, writer_pool, meter_pool, online_fit_meters, userID_last)					

		# adjust LR, copy optimal model, reset meter_pool, and print the optimum one here
		best_lr = get_lr(optim_pool[idx_best_model])
		writer.add_scalar("learning rate", best_lr, epoch)

		if (epoch + 1) % args.LR_adjust_intv == 0:
			print("[population_based_LR_adjust]: adjusting optimum lr...epoch = {}".format(epoch))
			population_based_LR_adjust(model_pool, writer_pool, optim_pool, meter_pool, idx_best_model, args)
		
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			save_dict = {
				'epoch': epoch + 1,
				'arch': args.arch,
				'size_replay_buffer': args.size_replay_buffer,
				'best_acc1': 0,
				'userID_last': userID_last,
				'online_fit_meters': online_fit_meters,
			}

			save_dict['state_dict'] = model_pool[1].state_dict()
			save_dict['optimizer'] = optim_pool[1].state_dict()

			for i in range(0, len(model_pool)):
				save_dict['state_dict{}'.format(i)] = model_pool[i].state_dict()
				save_dict['optimizer{}'.format(i)] = optim_pool[i].state_dict()
				save_dict['online_fit_meters{}'.format(i)] = meter_pool[i]

			if (epoch + 1) % 10 == 0 and args.SaveInter: 
				save_checkpoint(save_dict, 0, output_dir = args.output_dir, filename='checkpoint_ep{}.pth.tar'.format(epoch + 1))
			else:
				save_checkpoint(save_dict, 0, output_dir = args.output_dir)

	if args.use_val == 1:
		val_loader  = torch.utils.data.DataLoader(val_set,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
		acc1 = validate(val_loader, model_pool[idx_best_model], criterion, args, writer, args.epochs)
		print("final average accuracy = {}".format(acc1))

def find_next_test_album(user_ID, user_ID_last, idx_time_sorted):
	idx_ne = (user_ID[idx_time_sorted] != user_ID_last).nonzero().flatten()
	if idx_ne.numel() > 0:
		idx_end = 1
		for i in range(1, idx_ne.numel()):
			if user_ID[idx_time_sorted][idx_ne[0]] == user_ID[idx_time_sorted][idx_ne[i]]:
				idx_end += 1
			else:
				break
		return idx_ne[:idx_end] 
	else:
		return None

def init_local_meters():
	meter_local = {}
	meter_local['batch_time'] = AverageMeter('Time', ':6.3f')
	meter_local['data_time'] = AverageMeter('Data', ':6.3f')

	meter_local['losses'] = AverageMeter('Loss', ':.4e')
	
	meter_local['top1'] = AverageMeter('Acc@1', ':6.2f')
	meter_local['top5'] = AverageMeter('Acc@5', ':6.2f')
	
	meter_local['top1_future'] = AverageMeter('AccF@1', ':6.2f')
	meter_local['top5_future'] = AverageMeter('AccF@5', ':6.2f')

	meter_local['top1_Rep'] = AverageMeter('Acc_old@1', ':6.2f')
	meter_local['top5_Rep'] = AverageMeter('Acc_old@5', ':6.2f')

	return meter_local

def set_train(model_pool):
	for model in model_pool:
		model.train()

def set_zero_grad(optim_pool):
	for optim in optim_pool:
		optim.zero_grad()

def set_eval(model_pool):
	for model in model_pool:
		model.eval()

def train_MultiGD(train_loader, model_pool, criterion, optim_pool, epoch, args, writer, writer_pool, meter_pool, online_fit_meters, userID_last):
	# initialize a dictionary of meters
	meter_local = init_local_meters()

	progress = ProgressMeter(
		len(train_loader),
		[meter_local['batch_time'], meter_local['data_time'], meter_local['losses'], meter_local['top1'], meter_local['top1_future'], meter_local['top1_Rep'], online_fit_meters[1], online_fit_meters[2]],
		prefix="Epoch: [{}]".format(epoch))   

	# switch to train mode
	set_train(model_pool) 
	end = time.time()

	set_zero_grad(optim_pool) 

	ngpus = torch.cuda.device_count()
	print("ngpus = {}".format(ngpus))

	rate_rep_sample = args.sample_rate_from_buffer/(1-args.sample_rate_from_buffer) 
	idx_best_model = 1

	index_max = 0
	for i, (images, target, userID, time_taken, index, 
			images_pop_tmp, target_pop_tmp, index_pop_tmp, 
			images_from_buf_tmp, target_from_buf_tmp, index_buf_tmp) in enumerate(train_loader):
		# 1. images & target: test mini-batch (size = batch size)
		# 2. images_pop & target_pop: samples poped from validation buffer, used for training (size = batch size, set to "images & target" if batch_size*iter < val_buf_size, and we do normal SGD in this case)
		# 3. images_from_buf and target_from_buf: samples from replay buffer, used for training (size = batch size * sample rate * number of GD steps per iter)

		index_max = max(index_max, index.max())
		# measure data loading time
		meter_local['batch_time'].update(time.time() - end)
		
		iter_curr = epoch*len(train_loader)/(args.gradient_steps_per_batch*args.NOSubBatch)+i//(args.gradient_steps_per_batch*args.NOSubBatch)
		batch_size = target_pop_tmp.numel()

		if i == 0:
			batch_size_ori = batch_size

		if i % (args.gradient_steps_per_batch*args.NOSubBatch) == 0:
			images = images.reshape(images.size()[0]*images.size()[1],images.size()[2], images.size()[3], images.size()[4])[:]
			target = target.reshape(1, -1)[0][:]
			userID = userID.reshape(1, -1)[0][:]
			time_taken = time_taken.reshape(1, -1)[0][:]
			index = index.reshape(1, -1)[0][:]
		# organize the format of samples from the replay buffer
		bs_from_buf = round(batch_size*rate_rep_sample)

		images_pop = images_pop_tmp
		target_pop = target_pop_tmp
		index_pop = index_pop_tmp

		images_from_buf = images_from_buf_tmp.reshape(images_from_buf_tmp.size()[0]*images_from_buf_tmp.size()[1],images_from_buf_tmp.size()[2], images_from_buf_tmp.size()[3], images_from_buf_tmp.size()[4])[:bs_from_buf]
		target_from_buf = target_from_buf_tmp.reshape(1, -1)[0][:bs_from_buf]
		index_buf = index_buf_tmp.reshape(1, -1)[0][:bs_from_buf]

		if i % (args.gradient_steps_per_batch * args.NOSubBatch) == 0:
			# do prediction on the new batch
			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
				target = target.cuda(args.gpu, non_blocking=True)

			time_last, idx_best_model = test_on_model_pool(args, model_pool, images, target, userID, time_taken, meter_pool, online_fit_meters, meter_local, idx_best_model, userID_last, criterion)
			
		images = images_pop.cuda(args.gpu, non_blocking=True)
		target = target_pop.cuda(args.gpu, non_blocking=True)		

		# new code starts from here, do not support sub-batches for now	
		if args.sample_rate_from_buffer > 0 and iter_curr > 1:
			images_from_buf = images_from_buf.cuda(args.gpu, non_blocking=True)
			target_from_buf = target_from_buf.cuda(args.gpu, non_blocking=True)	
			images_merged, target_merged = merge_data_gpu(images_from_buf, target_from_buf, images, target, args)
			flag_merged = 1
		else:
			images_merged = images
			target_merged = target
			flag_merged = 0

		train_on_model_pool(args, model_pool, optim_pool, meter_pool, meter_local, criterion, images_merged, target_merged, target.numel(), idx_best_model, i)

		# measure elapsed time
		meter_local['batch_time'].update(time.time() - end)
		end = time.time()

		if i % (args.print_freq * args.gradient_steps_per_batch * args.NOSubBatch) == 0:
			progress.display(i)

		if i % (args.write_freq*args.gradient_steps_per_batch* args.NOSubBatch) == 0:
			write_tensor_board(meter_pool, meter_local, writer, writer_pool, online_fit_meters, iter_curr, time_last, extra_name = '_iter')
		
	# if args.gpu == 0:
	write_tensor_board(meter_pool, meter_local, writer, writer_pool, online_fit_meters, epoch, time_last)
	
	writer.add_scalar("RepBuf_size", args.size_replay_buffer, epoch)
	if args.use_ADRep:
		if meter_local['top1_Rep'].avg - meter_local['top1_future'].avg > args.ABS_performanceGap:
			args.size_replay_buffer	= int(args.size_replay_buffer*2)	
		elif meter_local['top1_Rep'].avg - meter_local['top1_future'].avg < -args.ABS_performanceGap:
			args.size_replay_buffer	= int(args.size_replay_buffer/2)

		print("[changing replay buffer size]: changing replay buffer size to {} (rep vs future = {}/{})".format(args.size_replay_buffer, meter_local['top1_Rep'].avg, meter_local['top1_future'].avg))	
		args.size_replay_buffer = min(index_max + 1, args.size_replay_buffer)
		print("[changing replay buffer size]: replay buffer size after min() = {}".format(args.size_replay_buffer))	

	return userID_last, idx_best_model

def test_on_model_pool(args, model_pool, images, target, userID, time_taken, meter_pool, online_fit_meters, meter_local, idx_best_model, userID_last, criterion):
	with torch.no_grad():
		set_eval(model_pool) 
		output_pool = [None] * len(model_pool)
		
		# don't put anything inside this for loop to avoid synchronization			
		for i in range(len(model_pool)):
			output_pool[i] = model_pool[i](images)
		
		target_new = global_gather(target)
		# compute online fit on the next album
		userID = global_gather(userID.cuda(args.gpu, non_blocking=True))
		time_taken = global_gather(time_taken.cuda(args.gpu, non_blocking=True))
		time_sorted, idx_sort = time_taken.sort()
		time_last = time_sorted[-1]
		idx_set = find_next_test_album(userID, userID_last, idx_sort)
		
		loss_best = 1e8
		idx_loss_best = 0
		loss_BCE_best = 1e8
		idx_loss_BCE_best = 0
		acc_best = -1.0
		idx_acc_best = 0
		acc5_best = -1.0
		idx_acc5_best = 0
		for i in range(len(model_pool)):
			output_all = global_gather(output_pool[i])
			acc1F, acc5F = accuracy(output_all, target_new, topk=(1, 5))

			meter_pool[i][1].update(acc1F[0])
			if acc_best < meter_pool[i][1].avg:
				idx_acc_best = i
				acc_best = meter_pool[i][1].avg

			if i == idx_best_model:
				meter_local['top1_future'].update(acc1F[0])
				meter_local['top5_future'].update(acc5F[0])

				if idx_set is not None:
					output_album = output_all[idx_sort][idx_set]
					target_album = target_new[idx_sort][idx_set]
					acc1OF, acc5OF = accuracy(output_album, target_album, topk=(1,5))
					online_fit_meters[1].update(acc1OF[0])
					online_fit_meters[2].update(acc5OF[0])
						
					# update userID_last
					userID_last = userID[idx_sort[-1]]

		set_train(model_pool)


	return time_last, idx_acc_best


def train_on_model_pool(args, model_pool, optim_pool, meter_pool, meter_local, criterion, images_merged, target_merged, size_new, idx_best_model, i_curr):
	output = [None] * len(model_pool)
	loss_pool = [None] * len(model_pool)
	for i in range(len(model_pool)):
		output[i] = model_pool[i](images_merged)
		loss_pool[i] = compute_loss(args, output[i], target_merged, criterion, size_new)
	
	for i in range(len(model_pool)):			
		loss_pool[i].backward()
		optim_pool[i].step()        # update parameters of net
		optim_pool[i].zero_grad()   # reset gradient

	if i_curr % (args.gradient_steps_per_batch * args.NOSubBatch) == 0:
		meter_local['losses'].update(loss_pool[idx_best_model].item(), target_merged.numel())
		acc1, acc5 = accuracy(output[idx_best_model], target_merged, topk=(1, 5))
		meter_local['top1'].update(acc1[0], target_merged.numel())
		meter_local['top5'].update(acc5[0], target_merged.numel())

		if target_merged.numel() > size_new:			
			output_all = output[idx_best_model][size_new:]	
			target_all = target_merged[size_new:]					
			acc1, acc5 = accuracy(output_all, target_all, topk=(1, 5))
			
			meter_local['top1_Rep'].update(acc1[0], target_all.numel())
			meter_local['top5_Rep'].update(acc5[0], target_all.numel())

def write_tensor_board(meter_pool, meter_local, writer, writer_pool, online_fit_meters, iter_curr, time_last, extra_name = ''):

		if extra_name == '':
			name_epoch = '_epoch'
		else:
			name_epoch = ''

		writer.add_scalar("train_loss{}".format(extra_name), meter_local['losses'].avg, iter_curr)
		writer.add_scalar("train_acc1{}".format(extra_name), meter_local['top1'].avg, iter_curr)
		writer.add_scalar("train_acc5{}".format(extra_name), meter_local['top5'].avg, iter_curr)

		writer.add_scalar("avg_online_loss{}".format(extra_name+name_epoch), online_fit_meters[0].avg, iter_curr)
		writer.add_scalar("avg_online_acc1{}".format(extra_name+name_epoch), online_fit_meters[1].avg, iter_curr)
		writer.add_scalar("avg_online_acc5{}".format(extra_name+name_epoch), online_fit_meters[2].avg, iter_curr)

		writer.add_scalar("avg_online_loss_time{}".format(extra_name), online_fit_meters[0].avg, time_last)
		writer.add_scalar("avg_online_acc1_time{}".format(extra_name), online_fit_meters[1].avg, time_last)
		writer.add_scalar("avg_online_acc5_time{}".format(extra_name), online_fit_meters[2].avg, time_last)

		writer.add_scalar("train_acc1_old{}".format(extra_name), meter_local['top1_Rep'].avg, iter_curr)
		writer.add_scalar("train_acc5_old{}".format(extra_name), meter_local['top5_Rep'].avg, iter_curr)
					
		writer.add_scalar("train_acc1_future{}".format(extra_name), meter_local['top1_future'].avg, iter_curr)
		writer.add_scalar("train_acc5_future{}".format(extra_name), meter_local['top5_future'].avg, iter_curr)
		
		for i in range(len(meter_pool)):
			writer_pool[i].add_scalar("train_loss_future{}".format(extra_name), meter_pool[i][0].avg, iter_curr)
			writer_pool[i].add_scalar("train_acc1_future{}".format(extra_name), meter_pool[i][1].avg, iter_curr)
			writer_pool[i].add_scalar("train_acc5_future{}".format(extra_name), meter_pool[i][2].avg, iter_curr)

# change this code later after we finish the code for training
def validate(val_loader, model, criterion, args, writer, epoch, is_forward = 2, time_last = -1):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')

	top1_iter = AverageMeter('AccI@1', ':6.2f')
	top5_iter = AverageMeter('AccI@5', ':6.2f')
	
	top1_over_time = AverageMeter('Acc@1_time', ':6.2f')
	top5_over_time = AverageMeter('Acc@5_time', ':6.2f')
	
	progress = ProgressMeter(
		len(val_loader),
		[batch_time, losses, top1, top5, top1_iter, top5_iter, top1_over_time, top5_over_time],
		prefix='Test: ')

	model.eval()

	week_per_second = 24*7*3600
	idx_gap_plot = 1e4
	idx_gap_number = 1
	time_init = time_last
	with torch.no_grad():
		end = time.time()
		for i, (images, target, time_curr, idx) in enumerate(val_loader):
			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
				target = target.cuda(args.gpu, non_blocking=True)
			
			if i == 0:
				if time_last == -1:
					time_last = time_curr[0]
				idx_init = idx[0]
			
			# compute output
			output = model(images)
			loss = criterion(output, target)
			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), target.size(0))
			top1.update(acc1[0], target.size(0))
			top5.update(acc5[0], target.size(0))

			top1_over_time.update(acc1[0], target.size(0))
			top5_over_time.update(acc5[0], target.size(0))
			top1_iter.update(acc1[0], target.size(0))
			top5_iter.update(acc5[0], target.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()


			if i % args.print_freq == 0:
				progress.display(i)
				sys.stdout.flush() 
			
			time_latest = time_curr[-1]
			idx_latest = idx[-1]
			if is_forward == 0:
				time_gap_from_init = time_init - time_latest
				time_gap = time_last - time_latest
				idx_gap =  idx_init - idx_latest
			else:
				time_gap_from_init = time_latest - time_init
				time_gap = time_latest - time_last 
				idx_gap =  idx_latest - idx_init 

			if idx_gap > idx_gap_number*idx_gap_plot or i == (len(val_loader)-1):   
				writer.add_scalar("val_acc1_iter", top1_iter.avg, idx_gap)                    
				writer.add_scalar("val_acc5_iter", top5_iter.avg, idx_gap)

				writer.add_scalar("transfer_top1_valIdx", top1.avg, idx_gap)                    
				writer.add_scalar("transfer_top5_valIdx", top5.avg, idx_gap)
				                    
				top1_iter.reset()
				top5_iter.reset()

				idx_gap_number += 1

			if time_gap > week_per_second:
				writer.add_scalar("transfer_top1_time", top1.avg, time_gap_from_init)                    
				writer.add_scalar("transfer_top5_time", top5.avg, time_gap_from_init)                    

				writer.add_scalar("val_acc1_over_time", top1_over_time.avg, time_gap_from_init)                    
				writer.add_scalar("val_acc5_over_time", top5_over_time.avg, time_gap_from_init)                    

				top1_over_time.reset()
				top5_over_time.reset()
				time_last = time_latest

		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	# if args.gpu == 0:
	writer.add_scalar("val_loss", losses.avg, epoch)
	writer.add_scalar("val_acc1", top1.avg, epoch)
	return top1.avg


def save_checkpoint(state, is_best, output_dir = '.', filename='checkpoint.pth.tar'):
	torch.save(state, output_dir+'/checkpoint.pth.tar')
	if filename != 'checkpoint.pth.tar':
		shutil.copyfile(output_dir+'/checkpoint.pth.tar', output_dir+'/'+filename)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def set(self, meter):
		self.val = meter.val
		self.avg = meter.avg
		self.sum = meter.sum
		self.count = meter.count

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		# for test only
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def create_output_dir(args):
	output_dir = 'results_best_model/'+args.arch+'_lr{}_epochs{}'.format(args.lr, args.epochs)+'_bufSize{}'.format(args.size_replay_buffer)    	
	
	if args.sampling_strategy != 'FIFO':
		output_dir += '_sampleStrategy{}'.format(args.sampling_strategy)

	if args.weight_old_data != 1.0:
		output_dir += '_weightOld{}'.format(args.weight_old_data)
	
	if args.ReplayType != 'mixRep':
		output_dir += '_{}'.format(args.ReplayType)

	if args.LR_adjust_intv != 1:
		output_dir += '_LrAjIntv{}'.format(args.LR_adjust_intv)		

	if args.ABS_performanceGap != 0.5:
		output_dir += 'ABSPG{}'.format(args.ABS_performanceGap)	
	
	if args.gradient_steps_per_batch > 1:
		 output_dir+='_GDSteps{}'.format(args.gradient_steps_per_batch)

	output_dir += '_BS{}'.format(args.batch_size)
	
	if args.GN > 0:
		output_dir += '_GN{}'.format(args.GN)

	if args.NOSubBatch > 1:
		output_dir += '_NOSubBatch{}'.format(args.NOSubBatch)

	if args.weight_decay != 1e-4:
		output_dir += '_WD{}'.format(args.weight_decay)

	if args.use_ADRep <=0:
		output_dir += '_noADRep'  

	if args.evaluate:
		output_dir += '/evaluate'    

	return output_dir



def global_gather(x):
	all_x = [torch.ones_like(x)
			 for _ in range(dist.get_world_size())]
	dist.all_gather(all_x, x, async_op=False)
	return torch.cat(all_x, dim=0)


def compute_loss_CE(args, output, target, criterion, size_subBatch_new):
	lossF = criterion(output[:size_subBatch_new], target[:size_subBatch_new])
	if target.numel() > size_subBatch_new  and args.weight_old_data > 0.0 and args.sample_rate_from_buffer > 0.0:
		lossRep = criterion(output[size_subBatch_new:], target[size_subBatch_new:])	
		return (lossF*0.5+args.weight_old_data*lossRep*0.5)
	else:
		return lossF


def compute_loss(args, output, target, criterion, size_subBatch_new):
	return compute_loss_CE(args, output, target, criterion, size_subBatch_new)


def merge_data(image_buf, target_buf, image, target):
	# currently just simply merge them
	if image_buf.is_cuda:
		image_buf = image_buf.cpu()
	if target_buf.is_cuda:
		target_buf = target_buf.cpu()

	return torch.cat((image, image_buf)), torch.cat((target, target_buf))

def merge_data_gpu(image_buf, target_buf, image, target, args):
	# currently just simply merge them
	if not image_buf.is_cuda:
		image_buf = image_buf.cuda(args.gpu, non_blocking=True)
	if not target_buf.is_cuda:
		target_buf = target_buf.cuda(args.gpu, non_blocking=True)

	return torch.cat((image, image_buf)), torch.cat((target, target_buf))


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']

def population_based_LR_adjust(model_pool, writer_pool, optim_pool, meter_pool, idx_best_model, args):
	metric_middle = meter_pool[1][1]
	metric_best = meter_pool[idx_best_model][1]

	if metric_best == metric_middle:
		idx_best_model = 1

	LR_min = args.LRMultiFactor * get_lr(optim_pool[idx_best_model])

	for i in range(len(model_pool)):
		if i != idx_best_model:
			model_pool[i] = copy.deepcopy(model_pool[idx_best_model])
		print("[reset meter pool]: meter_pool[{}][0] = {}; idx_best_model = {}".format(i, meter_pool[i][0].avg, idx_best_model))
		reset_meter_pool(meter_pool[i])
		optim_pool[i] = torch.optim.SGD(model_pool[i].parameters(), LR_min /float(args.LRMultiFactor**i),
									momentum=args.momentum,
									weight_decay=args.weight_decay)

		print("[model copy]: setting lr[{}] to {}; meter_pool[i] = {}".format(i, get_lr(optim_pool[i]), meter_pool[i]))


def copy_meter(meter_source, meter_target):
	for i in range(len(meter_source)):
		meter_target[i].set(meter_source[i])
	return meter_target

def reset_meter_pool(meter_pool):
	for i in range(len(meter_pool)):
		meter_pool[i].reset()
	
def set_lr(optim, lr):
	for param_group in optim.param_groups:
		param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args, ngpus_per_node, iter_curr = 0):
	lr = args.lr
	
	lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
	lr = max(args.min_lr, lr)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,), cross_GPU = False):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		if cross_GPU:
			output = global_gather(output)
			target = global_gather(target)
			
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.reshape(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res




if __name__ == '__main__':
	main()