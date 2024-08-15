import os
from pathlib import Path

class GlobalConfig:
	""" base architecture configurations """
	# Data
	seq_len = 1 # input timesteps
	pred_len = 4 # future waypoints predicted

	# data root
	root_dir_all = os.getenv("DATASET_DIR")

	train_towns = ['town01', 'town04', 'town10']
	val_towns = ['town02', 'town05', 'town07']

	train_data, val_data = [], []
	for town in train_towns:		
		train_data.append(os.path.join(root_dir_all, town))
		train_data.append(os.path.join(root_dir_all, town+'_addition'))
	for town in val_towns:
		val_data.append(os.path.join(root_dir_all, town+'_val'))

	ignore_sides = True # don't consider side cameras
	ignore_rear = True # don't consider rear cameras

	input_resolution = 256

	scale = 1 # image pre-processing
	crop = 256 # image pre-processing

	lr = 1e-4 # learning rate

	# Controller
	turn_KP = 0.75
	turn_KI = 0.75
	turn_KD = 0.3
	turn_n = 40 # buffer size

	speed_KP = 5.0
	speed_KI = 0.5
	speed_KD = 1.0
	speed_n = 40 # buffer size

	max_throttle = 0.75 # upper limit on throttle signal value in dataset
	brake_speed = 0.4 # desired speed below which brake is triggered
	brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
	clip_delta = 0.25 # maximum change in speed input to logitudinal controller


	aim_dist = 4.0 # distance to search around for aim point
	angle_thresh = 0.3 # outlier control detection angle
	dist_thresh = 10 # target point y-distance for outlier filtering


	speed_weight = 0.05
	value_weight = 0.001
	features_weight = 0.05

	rl_ckpt = "roach/log/ckpt_11833344.pth"

	img_aug = True


	def __init__(self, split=None, **kwargs):
		# split is a number between 0 and 5 indicating the index of towns.
		self.split = split
		if self.split is not None:
			if split < 0 or split > 5:
				raise Exception("There are only 6 towns to be left out. Please provide a number between 0 and 5.")
			else:
				towns = ['town01', 'town02', 'town04', 'town05', 'town07', 'town10']
				self.left_out_town = towns.pop(self.split)
				self.train_towns = towns
				self.val_towns = towns
				self.test_towns = [self.left_out_town]
			
			# Update self.train_data, self.val_data, self.test_data
			self.train_data, self.val_data, self.test_data = [], [], []
			for town in self.train_towns:
				self.train_data.append(os.path.join(self.root_dir_all, town))
				# Add town additional data if exists
				town_path_addition = Path(os.path.join(self.root_dir_all, town+'_addition'))
				if town_path_addition.exists():
					self.train_data.append(str(town_path_addition))
			for town in self.val_towns:
				self.val_data.append(os.path.join(self.root_dir_all, town+"_val"))
			for town in self.test_towns:
				self.test_data.append(os.path.join(self.root_dir_all, town+"_val"))

		for k,v in kwargs.items():
			setattr(self, k, v)
