from datetime import datetime
import sys
import csv

import io
import random
import shutil
import sys
from multiprocessing import Pool
import pathlib

import requests
from PIL import Image
import time
import os

import wget

def image_downloader(url_and_path:list):
	img_url = url_and_path[0]
	save_path = url_and_path[1]

	try:
		res = requests.get(img_url, stream=True)
		# count = 1
		# while res.status_code not in [ 301, 302, 303, 307, 308, 200 ] and count <= 5:
		# 	res = requests.get(img_url, stream=True)
		# 	# print(f'Retry: {count} {img_url}')
		# 	count += 1
		# checking the type for image
		if 'image' not in res.headers.get("content-type", ''):
		# print('ERROR: URL doesnot appear to be an image')
			return 0
		if not os.path.exists(save_path):	
			os.makedirs(os.path.dirname(save_path), exist_ok = True)
			i = Image.open(io.BytesIO(res.content))
			i.save(save_path)
		return 1

	except:
		return 0	

def test_function(number):
	return number*number

def run_downloader(process:int, urls_and_paths:list):

	pool = Pool()

	results = pool.imap_unordered(image_downloader, urls_and_paths)
	total_number = 0
	success_number = 0

	for r in results:
		total_number = total_number + 1
		if r == 1:
			success_number = success_number + 1

	return success_number, total_number

header = ['photoid', 'uid', 'unickname', 'datetaken', 'capturedevice', 'title', 'description', 'usertags','machinetags','longitude','latitude','accuracy', 'web page urls', 'downloadurl original', 'downloadurl medium size images (used)', 'local path']


# f_out = open('new_metadata.csv.zip', 'r')
csv.field_size_limit(sys.maxsize)

urls_and_paths = []success_number_total = 0

num_process = 5 # accelerate downloading using multiple processes
current_start = 0 # restart from the previous image id, can be used to resume download if there is a time limit constraint to run the downloader
batch_size = 200 # number of images to download simultaneously
part_length = 50000000 # if you have the budget to run multiple downloaders at the same time, you can divide the dataset into several parts and specify the number of images in each part in here
current_part = 0 # if you have the budget to run multiple downloaders at the same time, you can divide the dataset into several parts and specify the part to download in here
current_end = part_length*(current_part+1)
root_folder = 'dataset/' # name of the root folder for storing the images, remember to add '/' and change the corresponding dataset folder in the code

print("downloading files [{} , {}]".format(current_start, current_end))
sys.stdout.flush()
fname = '../release/yfcc100m_full_dataset_alt/download_link_and_locations.csv'
with open(fname) as f:
	csv_reader = csv.reader(f, delimiter=',')
	line_count = 0
	for row in csv_reader:
		line_count += 1
		
		if line_count < current_start:
			continue
		elif line_count >= current_end:
			break

		downloadurl_m = row[0]
		local_path = root_folder + row[1]
		
		if line_count % batch_size == 0 :
			# print('line count = {}'.format(line_count))
			# print('downloadurl_m = {}; local path = {}'.format(downloadurl_m, local_path))
			urls_and_paths += [[downloadurl_m, local_path]]
			success_number, _  = run_downloader(num_process, urls_and_paths)
			success_number_total += success_number
			urls_and_paths = []
			if line_count % 1000 == 0:
				print('success_rate = {}/{}'.format(success_number_total, line_count))
				sys.stdout.flush()
		else:
			urls_and_paths += [[downloadurl_m, local_path]]


	print(f'Processed {line_count} lines.')
