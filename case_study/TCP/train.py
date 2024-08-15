import argparse
import os
import numpy as np
from collections import OrderedDict

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Beta
torch.manual_seed(0)
random.seed(0)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger

from case_study.TCP.model import TCP
from case_study.TCP.data import CARLA_Data
from case_study.TCP.config import GlobalConfig


def my_own_loss(prop_eval, prop_reference_output, pred, comparison, loss_type="mse", offset=0.0):
	pre_cond = prop_eval.bool()
	# Condition inverted because otherwise we are not violating the post-condition
	if comparison == "<":
		post_cond = pred >= prop_reference_output
		prop_reference_output = prop_reference_output * (1-offset)
	elif comparison == "<=":
		post_cond = pred > prop_reference_output
		prop_reference_output = prop_reference_output * (1-offset)
	elif comparison == ">":
		post_cond = pred <= prop_reference_output
		prop_reference_output = prop_reference_output * (1+offset)
	elif comparison == ">=":
		post_cond = pred < prop_reference_output
		prop_reference_output = prop_reference_output * (1+offset)
	else:
		raise NotImplementedError("Comparison not implemented.")
	mask = pre_cond & post_cond
	
	if (mask == False).all():
		return torch.mean((prop_reference_output - prop_reference_output)), mask.count_nonzero().detach(), mask
	if loss_type == "mse":
		return torch.mean((pred[mask] - prop_reference_output[mask])**2), mask.count_nonzero().detach(), mask
	elif loss_type == "mae":
		return torch.mean(torch.abs(pred[mask] - prop_reference_output[mask])), mask.count_nonzero().detach(), mask
	else:
		raise NotImplementedError("Loss type not implemented.")


class TCP_planner(pl.LightningModule):
	def __init__(self, config, lr):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = TCP(config)
		self._load_weight()
		self.lr_mult = {}
		for prop in args.properties:
			self.lr_mult[prop] = 0
		self.violation_list = {}

	def _load_weight(self):
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		self.model.train()
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']

		gt_waypoints = batch['waypoints']

		pred = self.model(front_img, state, target_point)

		# Property loss
		env_props = args.properties
		all_prop_loss = []
		stop_prop_mask = torch.ones(batch['action_sigma'].shape[0], device=self.device)
		total_violations = 0
		for i in range(len(batch["prop_eval"])):
			prop_name = batch["prop_name"][i][0]
			if prop_name in env_props:
				prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i], batch["prop_reference_output"][i], pred[batch["prop_attribute"][i][0]], batch["prop_comparison"][i][0])
				if args.prop_loss:
					all_prop_loss.append((prop_name, prop_loss * args.lambda_pl))
					if prop_name.endswith("_v6"):
						stop_prop_mask *= (~mask).long()
				self.log(f"train_{prop_name}_loss", prop_loss.detach(), on_step=False, on_epoch=True)
				self.log(f"train_violation_count_{prop_name}", vc, on_step=False, on_epoch=True, reduce_fx=sum)
				total_violations += vc

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		if args.mask_loss:
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred) * stop_prop_mask.unsqueeze(dim=1)
		else:
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		
		# Loss construction
		loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
		self.log('train_loss_nopl', loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

		for p_name, p_loss in all_prop_loss:
			# Apply property loss
			if args.prop_loss:
				loss += self.lr_mult[p_name] * p_loss
			# Save losses for lambda update
			if p_name not in self.violation_list.keys():
				self.violation_list[p_name] = [p_loss.item()]
			else:
				self.violation_list[p_name].append(p_loss.item())

		self.log("train_total_violations", total_violations, on_step=False, on_epoch=True, reduce_fx=sum)
		self.log('train_action_loss', action_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_wp_loss_loss', wp_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_speed_loss', speed_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_value_loss', value_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_feature_loss', feature_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_future_feature_loss', future_feature_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_future_action_loss', future_action_loss.detach(), on_step=False, on_epoch=True)
		self.log('train_loss',loss.detach(), on_step=False, on_epoch=True)

		return loss

	def configure_optimizers(self):
		if args.optimizer == 'adam':
			optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		elif args.optimizer == 'sgd':
			optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lrs_step, args.lrs_cut)
		return [optimizer], [lr_scheduler]

	def validation_step(self, batch, batch_idx):
		self.model.eval()
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		gt_waypoints = batch['waypoints']

		pred = self.model(front_img, state, target_point)

		# Property loss
		env_props = args.properties
		all_prop_loss = []
		stop_prop_mask = torch.ones(batch['action_sigma'].shape[0], device=self.device)
		total_violations = 0
		for i in range(len(batch["prop_eval"])):
			prop_name = batch["prop_name"][i][0]
			if prop_name in env_props:
				prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i], batch["prop_reference_output"][i], pred[batch["prop_attribute"][i][0]], batch["prop_comparison"][i][0])
				if args.prop_loss:
					all_prop_loss.append(prop_loss * args.lambda_pl)
					if prop_name.endswith("_v6"):
						stop_prop_mask *= (~mask).long()
				self.log(f"val_{prop_name}_loss", prop_loss.detach(), on_step=False, on_epoch=True)
				self.log(f"val_violation_count_{prop_name}", vc, on_step=False, on_epoch=True, reduce_fx=sum)
				total_violations += vc

		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		if args.mask_loss:
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred) * stop_prop_mask.unsqueeze(dim=1)
		else:
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

		B = batch['action_mu'].shape[0]
		batch_steer_l1 = 0 
		batch_brake_l1 = 0
		batch_throttle_l1 = 0
		for i in range(B):
			throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
			batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])
			batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
			batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

		batch_throttle_l1 /= B
		batch_steer_l1 /= B
		batch_brake_l1 /= B

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len-1):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) *0.5 + torch.mean(kl_div[:, 1]) *0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len

		# Loss construction
		val_loss = wp_loss + batch_throttle_l1+5*batch_steer_l1+batch_brake_l1
		# val_loss = val_loss
		self.log('val_loss_nopl', val_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

		if args.prop_loss:
			for p_loss in all_prop_loss:
				val_loss += p_loss

		self.log("val_total_violations", total_violations, on_step=False, on_epoch=True, reduce_fx=sum)
		self.log("val_action_loss", action_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_speed_loss', speed_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_value_loss', value_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_feature_loss', feature_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_wp_loss_loss', wp_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_future_feature_loss', future_feature_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_future_action_loss', future_action_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)
		self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, sync_dist=True)

	def test_step(self, batch, batch_id):
		self.validation_step(batch, batch_id)

# Define a callback to retrieve learning rate
class LearningRateLoggerCallback(pl.Callback):
	def __init__(self, logger):
		super().__init__()
		self.logger = logger

	def on_epoch_start(self, trainer, pl_module):
		optimizer = trainer.optimizers[0]
		current_lr = optimizer.param_groups[0]['lr']
		logger.log_metrics({"learning_rate": current_lr}, step=trainer.current_epoch)
		print(f'Learning rate at epoch {trainer.current_epoch}: {current_lr}')

class PrintBiggestGradientCallback(pl.Callback):
	def __init__(self, logger):
		super().__init__()
		self.biggest_gradient_b = 0
		self.biggest_gradient_e = 0
		self.logger = logger

	def on_after_backward(self, trainer, pl_module):
        # Get the biggest gradient
		max_grad = -1
		for param in pl_module.parameters():
			if param.grad is not None:
				param_max_grad = param.grad.abs().max().item()
				if param_max_grad > max_grad:
					max_grad = param_max_grad
					self.biggest_gradient_b = max_grad
				if param_max_grad > self.biggest_gradient_e:
					self.biggest_gradient_e = max_grad

	def on_batch_end(self, trainer, pl_module):
		self.logger.log_metrics({"biggest_gradients_batch": self.biggest_gradient_b}, step=trainer.current_epoch * len(trainer.train_dataloader) + trainer.batch_idx)
	
	def on_epoch_end(self, trainer, pl_module):
		if self.biggest_gradient_e > 0:
			self.logger.log_metrics({"biggest_gradients_epoch": self.biggest_gradient_e}, step=trainer.current_epoch)
			self.biggest_gradient_e = 0

class UpdateLrMultCallBack(pl.Callback):
	def __init__(self, logger, args):
		super().__init__()
		self.logger = logger
		self.args = args

	def on_train_epoch_end(self, trainer, pl_module, outputs):
		if args.prop_loss:
			for p_name in pl_module.lr_mult.keys():
				pl_module.lr_mult[p_name] += self.args.lambda_mult * np.mean(pl_module.violation_list[p_name])
				self.logger.log_metrics({f"{p_name} loss lambda after Epoch": pl_module.lr_mult[p_name]}, step=trainer.current_epoch)
			pl_module.violation_list = {}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--lrs_step', type=int, default=30, help='Learning rate scheduler: step. E.g. 30 epochs')
	parser.add_argument('--lrs_cut', type=float, default=0.5, help='Learning rate scheduler: cut. E.g. 0.5')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
	parser.add_argument('--resume_from', type=str, default=None, help='resume from checkpoint')
	parser.add_argument('--prop_loss', action='store_true', help='Enable property loss')
	parser.add_argument('--mask_loss', action='store_true', help='Enable main property masking')
	parser.add_argument('--evaluate_model', action='store_true', help='Only evaluate model')
	parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader.')
	parser.add_argument('--lambda_pl', type=float, default=1.0, help='Property loss multiplier.')
	parser.add_argument('--properties', type=str, nargs='+', help='List of properties to check')
	parser.add_argument('--lambda_mult', type=float, default=1e-1, help='Property lambda multiplier.')
	parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to be used.')
	parser.add_argument('--augment_data', action='store_true', help='Enable augmentation during training.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	print(f"Experiment ID: {args.id}")

	# Config
	config = GlobalConfig()

	# Data
	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug, augment_data=args.augment_data)
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data, augment_data=args.augment_data)

	def seed_worker(worker_id): # https://pytorch.org/docs/stable/notes/randomness.html
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	num_workers = args.num_workers
	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=False)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=False)

	TCP_model = TCP_planner(config, args.lr)
	if args.resume_from is not None:
		ckpt = torch.load(args.resume_from, map_location=torch.device("cpu"))
		ckpt = ckpt["state_dict"]
		# Add missing keys from model
		for key in list(TCP_model.state_dict().keys())[-11:]:
			ckpt[key] = TCP_model.state_dict()[key]
		TCP_model.load_state_dict(ckpt, strict = False)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

	logger = TensorBoardLogger(save_dir=args.logdir, default_hp_metric=False)

	trainer = pl.Trainer.from_argparse_args(args,
										 	logger=logger,
											gpus = args.gpus,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback, LearningRateLoggerCallback(logger), PrintBiggestGradientCallback(logger), UpdateLrMultCallBack(logger, args)],
											check_val_every_n_epoch = args.val_every,
											max_epochs = args.epochs,
											num_sanity_val_steps=0,
											gradient_clip_val=1.0
											)

	trainer.test(TCP_model, dataloader_val)
	TCP_model.model.output_adapter.requires_grad_(False)
	if not args.evaluate_model:
		trainer.fit(TCP_model, dataloader_train, dataloader_val)
