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
						' (default: resnet18)')
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
parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
					help='number of gradient accumulation steps')

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

parser.add_argument('--cell_id', default='../final_metadata/cellID_yfcc100m_metadata_with_labels_usedDataRatio0.05_t110000_t250.npy', type=str,
					help='file that store the cell IDS')

parser.add_argument('--val_freq', default=10, 
					type=int, help='perform validation per [val_freq] epochs (in offline mode)')
parser.add_argument('--adjust_lr', default=0, 
					type=int, help='whether to adjust lr')
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
					type=int, help='size of the experience replay buffer (per gpu, so if you have 8 gpus, each gpu will have size_replay_buffer number of samples in the buffer)')
parser.add_argument('--sample_rate_from_buffer', default=0.5, 
					type=float, help='how many samples are from the buffer')
parser.add_argument('--sampling_strategy', default='FIFO', 
					type=str, help='sampling strategy (FIFO, Reservoir, RingBuf)')
parser.add_argument('--num_classes', default=500, 
					type=int, help='number of classes (used only for constructing ring buffer, no need to set this value)')

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
					choices=['mixRep', 'pureRep'], help='Type of replay buffer')
parser.add_argument("--min_lr", default = 0.0, type = float, help = 'minimum learning rate, used to cut the cosine LR')
# whether to save intermediate models
parser.add_argument("--SaveInter", default = 1, type = int, help = 'whether to save intermediate models')
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
	sys.stdout.flush() 
	

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
				args.start_epoch, _, _, _, _ = resume_online(args, model, optimizer) 
			else:
				raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

		print("do not perform training for evaluation mode, val_set = {}".format(val_set))
		sys.stdout.flush() 

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
	num_classes = cell_ids.size + 1 # remember to +1
	print("init with num_classes = {}".format(num_classes))

	if args.GN == 0:
		if args.pretrained:
			print("=> using pre-trained model '{}'".format(args.arch))
			model = models.__dict__[args.arch](pretrained=True)
		else:
			print("=> creating model '{}'".format(args.arch))
			model = models.__dict__[args.arch](num_classes = num_classes, norm_layer = nn.SyncBatchNorm)
	else:
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

def resume_online(args, model, optimizer):
	print("=> loading checkpoint '{}'".format(args.resume))

	if args.gpu is None:
		checkpoint = torch.load(args.resume)
	else:
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
	if 'reservoir_buffer' in checkpoint:
		reservoir_buffer = checkpoint['reservoir_buffer']
	else:
		reservoir_buffer = None

	args.size_replay_buffer = checkpoint['size_replay_buffer']
	return checkpoint['epoch'], checkpoint['best_acc1'], reservoir_buffer, online_fit_meters, userID_last 


def init_val_loader(args, val_set):
	return torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

def init_loader_v2(args, train_set, epoch):
	train_set.size_buf = args.size_replay_buffer
	train_set._change_data_range(epoch = epoch)

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = False)
	else:
		train_sampler = None
	
	batchSize4Loader = int(args.batch_size/args.NOSubBatch)
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=batchSize4Loader, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler)
	return train_loader, train_sampler


def init_online_fit_meters():
	return [AverageMeter('LossOF', ':.4e'), AverageMeter('AccOF@1', ':6.2f'), AverageMeter('AccOF@5', ':6.2f'), AverageMeter('AccF@1', ':6.2f'), AverageMeter('AccO@1', ':6.2f')]

def train_online(args, model, criterion, optimizer, train_set, val_set, ngpus_per_node, writer):
	# init metrics for online fit
	online_fit_meters = init_online_fit_meters()
	
	userID_last = -1

	if args.resume:
		if os.path.isfile(args.resume):
			args.start_epoch, _, reservoir_buffer, online_fit_meters, userID_last = resume_online(args, model, optimizer)
			if reservoir_buffer is not None:
				train_set.buf_last = reservoir_buffer.cpu()
		else:
			raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

	for epoch in range(args.start_epoch, args.epochs):
		train_loader, train_sampler = init_loader_v2(args, train_set, epoch)

		if args.distributed:
			train_sampler.set_epoch(0)
		
		if args.adjust_lr:
			adjust_learning_rate(optimizer, epoch, args, ngpus_per_node)

		writer.add_scalar("learning rate", get_lr(optimizer), epoch)

		user_ID_last = train_MultiGD(train_loader, model, criterion, optimizer, epoch, args, writer, online_fit_meters, userID_last)					
	
		if not args.multiprocessing_distributed or (args.multiprocessing_distributed
				and args.rank % ngpus_per_node == 0):
			save_dict = {
				'epoch': epoch + 1,
				'arch': args.arch,
				'size_replay_buffer': args.size_replay_buffer,
				'state_dict': model.state_dict(),
				'best_acc1': 0,
				'optimizer' : optimizer.state_dict(),
				'online_fit_meters': online_fit_meters,
				'userID_last': userID_last,
			}

			# if args.gpu == 0:
			if args.sampling_strategy == 'Reservoir':
				# store previous reservoir buffer
				save_dict['reservoir_buffer'] = train_set.buf_last
		
			if (epoch + 1) % 10 == 0 and args.SaveInter: 
				save_checkpoint(save_dict, 0, output_dir = args.output_dir, filename='checkpoint_ep{}.pth.tar'.format(epoch + 1))
			else:
				save_checkpoint(save_dict, 0, output_dir = args.output_dir)

	if args.use_val == 1:
		val_loader  = torch.utils.data.DataLoader(val_set,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
		acc1 = validate(val_loader, model, criterion, args, writer, args.epochs)
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

def train_MultiGD(train_loader, model, criterion, optimizer, epoch, args, writer, online_fit_meters, userID_last):
	# these statistics are local, so we need a separate set of meters for online fit, forward and backward transfer.
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')

	losses = AverageMeter('Loss', ':.4e')

	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')

	top1_future = AverageMeter('AccF@1', ':6.2f')
	top5_future = AverageMeter('AccF@5', ':6.2f')

	top1_Rep = AverageMeter('Acc_old@1', ':6.2f')
	top5_Rep = AverageMeter('Acc_old@5', ':6.2f')

	progress = ProgressMeter(
		len(train_loader),
		[batch_time, data_time, losses, top1, top1_future, top1_Rep, online_fit_meters[1], online_fit_meters[2]],
		prefix="Epoch: [{}]".format(epoch))   

	# switch to train mode
	model.train()
	end = time.time()

	optimizer.zero_grad()

	ngpus = torch.cuda.device_count()

	rate_rep_sample = args.sample_rate_from_buffer/(1-args.sample_rate_from_buffer)

	for i, (images, target, userID, time_taken, index, 
			images_pop_tmp, target_pop_tmp, index_pop_tmp, 
			images_from_buf_tmp, target_from_buf_tmp, index_buf_tmp) in enumerate(train_loader):
		# 1. images & target: test mini-batch (size = batch size)
		# 2. images_pop & target_pop: samples poped from validation buffer, used for training (size = batch size, set to "images & target" if batch_size*iter < val_buf_size, and we do normal SGD in this case)
		# 3. images_from_buf and target_from_buf: samples from replay buffer, used for training (size = batch size * sample rate * number of GD steps per iter)

		# measure data loading time
		data_time.update(time.time() - end)
		
		# print("target = {}".format(target))
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

			with torch.no_grad():
				model.eval()
				output = model(images)
				output_all = global_gather(output)
				target_new = global_gather(target)

				acc1F, acc5F = accuracy(output_all, target_new, topk=(1, 5))
				top1_future.update(acc1F[0])
				top5_future.update(acc5F[0])

				# compute online fit on the next album
				userID = global_gather(userID.cuda(args.gpu, non_blocking=True))
				time_taken = global_gather(time_taken.cuda(args.gpu, non_blocking=True))
				time_sorted, idx_sort = time_taken.sort()
				time_last = time_sorted[-1]
				idx_set = find_next_test_album(userID, userID_last, idx_sort)
				if idx_set is not None:
					output_album = output_all[idx_sort][idx_set]
					target_album = target_new[idx_sort][idx_set]
					acc1OF, acc5OF = accuracy(output_album, target_album, topk=(1,5))
					online_fit_meters[1].update(acc1OF[0])
					online_fit_meters[2].update(acc5OF[0])
				
					# update userID_last
					userID_last = userID[idx_sort[-1]]	
						
				model.train()

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

		output = model(images_merged)
		loss = compute_loss(args, output, target_merged, criterion, target.numel())


		loss.backward()


		optimizer.step()        # update parameters of net
		optimizer.zero_grad()   # reset gradient

		if i % (args.gradient_steps_per_batch * args.NOSubBatch) == 0:

			losses.update(loss.item(), target_merged.numel())
			output_all = global_gather(output)
			target_all = global_gather(target_merged)
			acc1, acc5 = accuracy(output_all, target_all, topk=(1, 5))
			top1.update(acc1[0], target_all.numel())
			top5.update(acc5[0], target_all.numel())

			if target_merged.numel() > target.numel():			
				output_all = global_gather(output[target.numel():])	
				target_all = global_gather(target_merged[target.numel():])					
				acc1, acc5 = accuracy(output_all, target_all, topk=(1, 5))
				
				top1_Rep.update(acc1[0], target_all.numel())
				top5_Rep.update(acc5[0], target_all.numel())

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % (args.print_freq * args.gradient_steps_per_batch * args.NOSubBatch) == 0:
			progress.display(i)
			# sys.stdout.flush() 

		if i % (args.write_freq*args.gradient_steps_per_batch* args.NOSubBatch) == 0:
			writer.add_scalar("train_loss_iter", losses.avg, iter_curr)
			writer.add_scalar("train_acc1_iter", top1.avg, iter_curr)
			writer.add_scalar("train_acc5_iter", top5.avg, iter_curr)

			writer.add_scalar("avg_online_loss_iter", online_fit_meters[0].avg, iter_curr)
			writer.add_scalar("avg_online_acc1_iter", online_fit_meters[1].avg, iter_curr)
			writer.add_scalar("avg_online_acc5_iter", online_fit_meters[2].avg, iter_curr)

			writer.add_scalar("avg_online_loss_time_iter", online_fit_meters[0].avg, time_last)
			writer.add_scalar("avg_online_acc1_time_iter", online_fit_meters[1].avg, time_last)
			writer.add_scalar("avg_online_acc5_time_iter", online_fit_meters[2].avg, time_last)

			writer.add_scalar("train_acc1_old_iter", top1_Rep.avg, iter_curr)
			writer.add_scalar("train_acc5_old_iter", top5_Rep.avg, iter_curr)
						
			writer.add_scalar("train_acc1_future_iter", top1_future.avg, iter_curr)
			writer.add_scalar("train_acc5_future_iter", top5_future.avg, iter_curr)

				
	# if args.gpu == 0:
	writer.add_scalar("train_loss", losses.avg, epoch)
	writer.add_scalar("train_acc1", top1.avg, epoch)	
	writer.add_scalar("train_acc5", top5.avg, epoch)	
	
	writer.add_scalar("avg_online_loss_time", online_fit_meters[0].avg, time_last)
	writer.add_scalar("avg_online_acc1_time", online_fit_meters[1].avg, time_last)
	writer.add_scalar("avg_online_acc5_time", online_fit_meters[2].avg, time_last)

	writer.add_scalar("avg_online_loss_epoch", online_fit_meters[0].avg, epoch)
	writer.add_scalar("avg_online_acc1_epoch", online_fit_meters[1].avg, epoch)
	writer.add_scalar("avg_online_acc5_epoch", online_fit_meters[2].avg, epoch)
			
	writer.add_scalar("train_acc1_old", top1_Rep.avg, epoch)	
	writer.add_scalar("train_acc5_old", top5_Rep.avg, epoch)	
	
	writer.add_scalar("train_acc1_future", top1_future.avg, epoch)
	writer.add_scalar("train_acc5_future", top5_future.avg, epoch)
	
	writer.add_scalar("RepBuf_size", args.size_replay_buffer, epoch)
	if args.use_ADRep:
		if top1_Rep.avg - top1_future.avg > args.ABS_performanceGap:
			args.size_replay_buffer	= int(args.size_replay_buffer*2)	
		elif top1_Rep.avg - top1_future.avg < -args.ABS_performanceGap:
			args.size_replay_buffer	= int(args.size_replay_buffer/2)
		print("[changing replay buffer size]: changing replay buffer size to {} (rep vs future = {}/{})".format(args.size_replay_buffer, top1_Rep.avg, top1_future.avg))


	return userID_last

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

			if idx_gap % 10000 == 0 or i == (len(val_loader)-1):   
				writer.add_scalar("val_acc1_iter", top1_iter.avg, idx_gap)                    
				writer.add_scalar("val_acc5_iter", top5_iter.avg, idx_gap)

				writer.add_scalar("transfer_top1_valIdx", top1.avg, idx_gap)                    
				writer.add_scalar("transfer_top5_valIdx", top5.avg, idx_gap)
				                    
				top1_iter.reset()
				top5_iter.reset()

			if time_gap > week_per_second:
				writer.add_scalar("transfer_top1_time", top1.avg, time_gap_from_init)                    
				writer.add_scalar("transfer_top5_time", top5.avg, time_gap_from_init)                    

				writer.add_scalar("val_acc1_over_time", top1_over_time.avg, time_gap_from_init)                    
				writer.add_scalar("val_acc5_over_time", top5_over_time.avg, time_gap_from_init)                    

				top1_over_time.reset()
				top5_over_time.reset()
				time_last = time_latest

		# TODO: this should also be done with the ProgressMeter
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

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def ma_update(self, val, weight):
		print("[ma_update before]: avg = {}; weight = {}; val = {}".format(self.avg, weight, val))
		if weight >= 1.0:
			self.update(val)
		else:
			self.val = val
			self.avg = self.avg * weight + val * (1 - weight)
		print("[ma_update after]: avg = {}".format(self.avg))
		
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
	output_dir = 'results_no_PoLRS/'+args.arch+'_lr{}_epochs{}'.format(args.lr, args.epochs)+'_bufSize{}'.format(args.size_replay_buffer)    	

	if args.sampling_strategy != 'FIFO':
		output_dir += '_sampleStrategy{}'.format(args.sampling_strategy)

	if args.adjust_lr != 0:
		output_dir += '_adjustLr{}'.format(args.adjust_lr)
		if args.adjust_lr == 2:
			output_dir += '_MinLr{}'.format(args.cyclic_min_lr)
	
	if args.ReplayType != 'mixRep':
		output_dir += '_{}'.format(args.ReplayType)

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
	
	if args.min_lr > 0.0:
		output_dir += '_minLr{}'.format(args.min_lr)

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
	if not image_buf.is_cuda:
		image_buf = image_buf.cuda(args.gpu, non_blocking=True)
	if not target_buf.is_cuda:
		target_buf = target_buf.cuda(args.gpu, non_blocking=True)

	return torch.cat((image, image_buf)), torch.cat((target, target_buf))

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


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