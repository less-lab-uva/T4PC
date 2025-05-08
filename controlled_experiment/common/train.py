import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import dotenv
import pandas as pd
import json
import shutil
import time
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from controlled_experiment.base_models.resnet_model import ResnetModel
from controlled_experiment.base_models.vit_model import VitModel
from controlled_experiment.base_models.data import CARLA_Data
from TCP.config import GlobalConfig

# Use the .env file to load environment variables
dotenv.load_dotenv('.env', override=True)

class MyLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.json_log_paths = {
            "train": f"{log_dir}/train_logs.json",
            "validation": f"{log_dir}/validation_logs.json",
            "test": f"{log_dir}/test_logs.json"
        }
        # Create json files for the different splits
        empty_json = {}
        for _, split_path in self.json_log_paths.items():
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(empty_json, f)
        # Initialize Tensorboard SummaryWriter
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, *args, **kwargs):
        self.tensorboard_writer.add_scalar(*args, **kwargs)

    def close(self, *args, **kwargs):
        self.tensorboard_writer.close()

    def log(self, avg_total_steering_loss, avg_total_acceleration_loss, avg_total_loss, avg_total_loss_no_pl,
            prop_count_dict, prop_loss_dict, new_prop_loss_dict, epoch, split, batch_times, batch_main_loss_times,
                    batch_prop_loss_times, data):
        new_entry = {
            "avg_total_steering_loss": avg_total_steering_loss,
            "avg_total_acceleration_loss": avg_total_acceleration_loss,
            "avg_total_loss": avg_total_loss,
            "avg_total_loss_no_pl": avg_total_loss_no_pl,
            "prop_violation_count_dict": prop_count_dict,
            "prop_loss_dict": prop_loss_dict,
            "new_prop_loss_dict": new_prop_loss_dict,
            "batch_times": batch_times,
            "batch_main_loss_times": batch_main_loss_times,
            "batch_prop_loss_times": batch_prop_loss_times
        }
        with open(self.json_log_paths[split], "r", encoding="utf-8") as f:
            json_data = json.load(f)
            json_data[epoch] = new_entry
        with open(self.json_log_paths[split], "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        if split == "test" and data != {}:
            df = pd.DataFrame(data)
            df.to_csv(f"{self.log_dir}/violations/test__e{epoch}.csv", index=False)


def init_summary_writer(log_dir, reset_version_dir=False):
    """
    Initialize an instance of MyLogger with a Tensorboard SummaryWriter.

    Parameters:
    - log_dir (str): The base directory for Tensorboard logs.
    - reset_version_dir (bool): Whether to reset the version directory.

    Returns:
    - writer (MyLogger): MyLogger instance with Tensorboard SummaryWriter.
    """
    logdir_path = Path(log_dir)

    # Create the log_dir if it does not exist
    logdir_path.mkdir(parents=True, exist_ok=True)

    # Scan existing version folders
    existing_versions = [d for d in os.listdir(log_dir) if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]

    # Determine the next version number
    if not existing_versions:
        version_number = 0
    else:
        if reset_version_dir:
            version_number = 0
            if (logdir_path / "version_0").exists():
                shutil.rmtree(os.path.join(log_dir, "version_0"))
        else:
            version_numbers = [int(d.split('_')[1]) for d in existing_versions]
            version_number = max(version_numbers) + 1

    # Create the next version folder
    next_version_folder = os.path.join(log_dir, f'version_{version_number}')
    os.makedirs(next_version_folder, exist_ok=True)

    # Create violations folder for test csv files
    os.makedirs(f"{next_version_folder}/violations", exist_ok=True)

    writer = MyLogger(next_version_folder)

    return writer


def my_own_loss(prop_eval, prop_reference_output, pred, comparison, loss_type="mae", offset=0.0):
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
        # Return 0 loss if no violations
        loss1 = torch.mean((prop_reference_output - prop_reference_output))
        return loss1, loss1, mask.count_nonzero().item(), mask
    if loss_type == "mse":
        # Two losses:
        # 1. MSE between pred and prop_reference_output divided by # violations
        loss1 = torch.mean((pred[mask] - prop_reference_output[mask])**2)
        # 2. MSE between pred and prop_reference_output divided by all data points
        loss2 = torch.sum((pred[mask] - prop_reference_output[mask])**2) / len(pred)
        return loss1, loss2, mask.count_nonzero().item(), mask
    elif loss_type == "mae":
        # Two losses:
        # 1. Absolute difference between pred and prop_reference_output divided by # violations
        loss1 = torch.mean(torch.abs(pred[mask] - prop_reference_output[mask]))
        # 2. Absolute difference between pred and prop_reference_output divided by all data points
        loss2 = torch.sum(torch.abs(pred[mask] - prop_reference_output[mask])) / len(pred)
        return loss1, loss2, mask.count_nonzero().item(), mask
    else:
        raise NotImplementedError("Loss type not implemented.")


def compute_loss(batch, acceleration, steering_angle, outputs, device,
                 prop_loss_dict, new_prop_loss_dict, prop_count_dict, split, args, data={}):
    all_prop_loss = []
    all_new_prop_loss = []
    acc_prop_mask = torch.ones(acceleration.shape[0], device=device)
    steer_prop_mask = torch.ones(steering_angle.shape[0], device=device)

    # Property Losses
    prop_loss_start_time = time.time()
    for i in range(len(batch["prop_eval"])):
        prop_name = batch["prop_name"][i][0]
        if prop_name in args.properties:
            if batch["prop_attribute"][i][0] == "acceleration":
                prop_attribute = "acceleration_continuous"
            else:
                prop_attribute = "steering_angle"

            #TODO: It is using acceleration/steering_angle (ground truth) because it masks main loss for data points for which the label is wrong. Can we do this in the dataloader?
            # Use wrong labels to mask main loss
            if prop_attribute == "acceleration_continuous":
                _, _, _, mask = my_own_loss(batch["prop_eval"][i].to(device),
                                         batch["prop_reference_output"][i].to(
                                             device),
                                         acceleration,
                                         batch["prop_comparison"][i][0])
                acc_prop_mask *= (~mask).long()
            elif prop_attribute == "steering_angle":
                _, _, _, mask = my_own_loss(batch["prop_eval"][i].to(device),
                                         batch["prop_reference_output"][i].to(
                                             device),
                                         steering_angle,
                                         batch["prop_comparison"][i][0])
                steer_prop_mask *= (~mask).long()
            else:
                raise NotImplementedError("No mask for property attribute")

            prop_loss, new_prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i].to(device),
                                              batch["prop_reference_output"][i].to(
                                                  device),
                                              outputs[prop_attribute],
                                              batch["prop_comparison"][i][0])

            all_prop_loss.append((prop_name, prop_loss * args.lambda_pl))
            prop_loss_dict[f"{split}_{prop_name}_loss"] += prop_loss.item()
            all_new_prop_loss.append((prop_name, new_prop_loss * args.lambda_pl))
            new_prop_loss_dict[f"{split}_{prop_name}_loss"] += new_prop_loss.item()
            prop_count_dict[f"{split}_violation_count_{prop_name}"] += vc

            if split == "test":
                data["violation"].extend(mask.cpu().tolist())
                data["img_id"].extend(list(batch['img_dir'][0]))
                data["acc_label"].extend(batch['acceleration'].tolist())
                data["acc_prop_mask"].extend(acc_prop_mask.tolist())
                data["steer_label"].extend(batch['steer'].tolist())
                data["steer_prop_mask"].extend(steer_prop_mask.tolist())
                data["prop_name"].extend([prop_name] * len(mask))
                data["acc_pred"].extend(outputs["acceleration_continuous"].cpu().tolist())
                data["steer_pred"].extend(outputs["steering_angle"].cpu().tolist())
    prop_loss_time = time.time() - prop_loss_start_time

    # Main Losses
    main_loss_start_time = time.time()
    if args.mask_loss:
        acceleration_loss = F.mse_loss(
            outputs["acceleration_continuous"]*acc_prop_mask, acceleration*acc_prop_mask)
        if acc_prop_mask.count_nonzero().item() > 0:
            acceleration_loss *= len(acc_prop_mask) / \
                acc_prop_mask.count_nonzero().item()
        steering_angle_loss = F.mse_loss(
            outputs["steering_angle"]*steer_prop_mask, steering_angle*steer_prop_mask)
        if steer_prop_mask.count_nonzero().item() > 0:
            steering_angle_loss *= len(steer_prop_mask) / \
                steer_prop_mask.count_nonzero().item()
    else:
        acceleration_loss = F.mse_loss(
            outputs["acceleration_continuous"], acceleration)
        steering_angle_loss = F.mse_loss(
            outputs["steering_angle"], steering_angle)
    main_loss_time = time.time() - main_loss_start_time

    return acceleration_loss, steering_angle_loss, all_prop_loss, all_new_prop_loss, prop_loss_dict, new_prop_loss_dict, prop_count_dict, data, prop_loss_time, main_loss_time


def log_information(writer: MyLogger, total_loss, total_loss_no_pl, total_steering_loss, total_acceleration_loss,
                    prop_count_dict, prop_loss_dict, new_prop_loss_dict, split_loader, epoch, split, batch_times, batch_main_loss_times,
                    batch_prop_loss_times, data=None):
    # Calculations
    avg_total_steering_loss = total_steering_loss / len(split_loader)
    avg_total_acceleration_loss = total_acceleration_loss / len(split_loader)
    avg_total_loss = total_loss / len(split_loader)
    avg_total_loss_no_pl = total_loss_no_pl / len(split_loader)

    # MyLogger
    writer.log(avg_total_steering_loss, avg_total_acceleration_loss, avg_total_loss, avg_total_loss_no_pl, 
               prop_count_dict, prop_loss_dict, new_prop_loss_dict, epoch, split, batch_times, batch_main_loss_times,
                    batch_prop_loss_times, data)

    # Tensorboard
    for key, val in prop_loss_dict.items():
        writer.add_scalar(key, val / len(split_loader), epoch)
    for key, val in new_prop_loss_dict.items():
        writer.add_scalar(key + " (NEW!)", val / len(split_loader), epoch)
    for key, val in prop_count_dict.items():
        writer.add_scalar(key, val, epoch)
    writer.add_scalar(f'{split} Steer Loss (Epoch)', avg_total_steering_loss, epoch)
    writer.add_scalar(f'{split} Acceleration Loss (Epoch)', avg_total_acceleration_loss, epoch)
    writer.add_scalar(f'{split} Loss (Epoch)', avg_total_loss, epoch)
    writer.add_scalar(f'{split} Loss No PL (Epoch)', avg_total_loss_no_pl, epoch)


################
### Training ###
################

def train_one_epoch(model, train_loader, optimizer, device, epoch, writer, lr_mult, args):
    model.train()
    total_loss = 0.0
    total_loss_no_pl = 0.0
    total_steering_loss = 0.0
    total_acceleration_loss = 0.0
    prop_loss_dict = {}
    new_prop_loss_dict = {}
    prop_count_dict = {}
    split = "train"
    for prop_name in args.properties:
        prop_loss_dict[f"{split}_{prop_name}_loss"] = 0.0
        new_prop_loss_dict[f"{split}_{prop_name}_loss"] = 0.0
        prop_count_dict[f"{split}_violation_count_{prop_name}"] = 0.0

    # Wrap the train_loader with tqdm for the progress bar
    tqdm_train_loader = tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=True)
    violation_list = {}
    batch_times = []
    batch_main_loss_times = []
    batch_prop_loss_times = []

    for batch in tqdm_train_loader:
        inputs = batch['front_img'].to(device)
        acceleration = batch['acceleration'].to(device)
        steering_angle = batch['steer'].to(device)

        time_start = time.time()

        optimizer.zero_grad()
        # [acceleration, steer, brake, acceleration_continuous]
        outputs = model(inputs)

        # Compute loss
        acceleration_loss, steering_angle_loss, all_prop_loss, all_new_prop_loss, prop_loss_dict, new_prop_loss_dict, prop_count_dict, data, prop_loss_time, main_loss_time = compute_loss(
            batch, acceleration, steering_angle, outputs, device, prop_loss_dict, new_prop_loss_dict,
            prop_count_dict, split, args)

        total_acceleration_loss += acceleration_loss.item()
        total_steering_loss += steering_angle_loss.item()
        losses = acceleration_loss + steering_angle_loss
        total_loss_no_pl += losses.item()

        for idx, (p_name, p_loss) in enumerate(all_prop_loss):
            # Apply property loss
            if args.prop_loss:
                if args.new_prop_loss:
                    losses += lr_mult[p_name] * all_new_prop_loss[idx][1]
                else:
                    losses += lr_mult[p_name] * p_loss
            # Save losses for lambda update
            if p_name not in violation_list.keys():
                violation_list[p_name] = [p_loss.item()]
            else:
                violation_list[p_name].append(p_loss.item())

        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        total_time = time.time() - time_start
        if args.prop_loss:
            batch_times.append(total_time)
        else:
            # Subtract time for computing the loss
            batch_times.append(total_time - prop_loss_time)
        batch_main_loss_times.append(main_loss_time)
        batch_prop_loss_times.append(prop_loss_time)

    # Log information
    log_information(writer, total_loss, total_loss_no_pl, total_steering_loss, total_acceleration_loss,
                    prop_count_dict, prop_loss_dict, new_prop_loss_dict, train_loader, epoch, split, batch_times, batch_main_loss_times,
                    batch_prop_loss_times)

    # Update lr multiplier
    for p_name, p_val in lr_mult.items():
        lr_mult[p_name] += args.lambda_mult * np.mean(violation_list[p_name])
        writer.add_scalar(f'{p_name} loss lambda after (Epoch)', p_val, epoch)

    return lr_mult


##################
### Validation ###
##################

def validate_one_epoch(model, loader, device, epoch, writer, split, args):
    model.eval()
    total_loss = 0.0
    total_loss_no_pl = 0.0
    total_steering_loss = 0.0
    total_acceleration_loss = 0.0
    prop_loss_dict = {}
    new_prop_loss_dict = {}
    prop_count_dict = {}
    for prop_name in args.properties:
        prop_loss_dict[f"{split}_{prop_name}_loss"] = 0.0
        new_prop_loss_dict[f"{split}_{prop_name}_loss"] = 0.0
        prop_count_dict[f"{split}_violation_count_{prop_name}"] = 0.0

    # Wrap the loader with tqdm for the progress bar
    tqdm_loader = tqdm(loader, desc=f'{split.capitalize()} Epoch {epoch}', leave=True)
    batch_times = []
    batch_main_loss_times = []
    batch_prop_loss_times = []

    if split == "test":
        data = {
            "img_id":[],
            "acc_label":[],
            "acc_prop_mask":[],
            "steer_label":[],
            "steer_prop_mask":[],
            "prop_name":[],
            "violation":[],
            "acc_pred":[],
            "steer_pred":[],
        }
    else:
        data = {}

    with torch.no_grad():
        for batch in tqdm_loader:
            inputs = batch['front_img'].to(device)
            acceleration = batch['acceleration'].to(device)
            steering_angle = batch['steer'].to(device)

            start_time = time.time()
            # [acceleration, steer, brake, acceleration_continuous]
            outputs = model(inputs)

            # Compute loss
            acceleration_loss, steering_angle_loss, all_prop_loss, all_new_prop_loss, prop_loss_dict, new_prop_loss_dict, prop_count_dict, data, prop_loss_time, main_loss_time = compute_loss(
                batch, acceleration, steering_angle, outputs, device, prop_loss_dict, new_prop_loss_dict,
                prop_count_dict, split, args, data=data)

            total_acceleration_loss += acceleration_loss.item()
            total_steering_loss += steering_angle_loss.item()
            losses = acceleration_loss + steering_angle_loss
            total_loss_no_pl += losses.item()

            if args.prop_loss:
                for idx, (_, p_loss) in enumerate(all_prop_loss):
                    if args.new_prop_loss:
                        losses += all_new_prop_loss[idx][1]
                    else:
                        losses += p_loss

            total_loss += losses.item()

            total_time = time.time() - start_time
            if args.prop_loss:
                batch_times.append(total_time)
            else:
                # Subtract time for computing the loss
                batch_times.append(total_time - prop_loss_time)
            batch_main_loss_times.append(main_loss_time)
            batch_prop_loss_times.append(prop_loss_time)

    # Log information
    log_information(writer, total_loss, total_loss_no_pl, total_steering_loss, total_acceleration_loss,
                    prop_count_dict, prop_loss_dict, new_prop_loss_dict, loader, epoch, split, batch_times, batch_main_loss_times,
                    batch_prop_loss_times, data=data)

    avg_total_loss = total_loss / len(loader)

    return avg_total_loss

############
### Main ###
############

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed used: {args.seed}")
    else:
        torch.manual_seed(0)
        print("Seed used: 0")

    args.log_dir = os.path.join(args.log_dir, args.id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloaders
    config = GlobalConfig(split=args.split_n)

    # Train Dataset
    train_dataset = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug,
                               transform_for=args.model_type, data_id="packed_data_ce.npy")
    print(f"Length of train: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, persistent_workers=True)

    # Validation Dataset
    val_dataset = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,
                             transform_for=args.model_type, data_id="packed_data_ce.npy")
    print(f"Length of val: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, persistent_workers=True)

    # Test Dataset
    test_dataset = CARLA_Data(root=config.root_dir_all, data_folders=config.test_data,
                             transform_for=args.model_type, data_id="packed_data_ce.npy")
    print(f"Length of test: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, persistent_workers=True)

    # Initialize model
    if args.model_type == 'vit':
        model = VitModel()
    else:
        model = ResnetModel()

    # Load weights
    if args.model_ckpt is not None:
        model.load_state_dict(torch.load(args.model_ckpt, map_location=torch.device('cpu')))
        print(f"Model loaded from {args.model_ckpt}")
    model.to(device)

    optimizer = None
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("Optimizer not supported.")

    # Set up learning rate scheduler
    scheduler = None
    if args.lrscheduler:
        scheduler = ReduceLROnPlateau(optimizer, patience=3, threshold=1e-3)

    # MyLogger with Tensorboard writer
    writer = init_summary_writer(args.log_dir, args.reset_version_dir)

    # Loop
    lr_mult = {}
    for prop in args.properties:
        lr_mult[prop] = 0
    for epoch in range(0, args.epochs):
        lr_mult = train_one_epoch(model, train_loader, optimizer, device, epoch, writer, lr_mult, args)

        if (epoch+1) % args.val_every == 0:
            # Log val data
            val_loss = validate_one_epoch(model, val_loader, device, epoch, writer, "validation", args)
            # Update lr scheduler
            if scheduler is not None:
                before_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                if optimizer.param_groups[0]["lr"] != before_lr:
                    print(f"Epoch {epoch}: lr updated to {optimizer.param_groups[0]['lr']}")
            # Log test data
            _ = validate_one_epoch(model, test_loader, device, epoch, writer, "test", args)

        if (epoch+1) % args.test_every == 0:
            # Save model
            ckpt_dir = writer.log_dir + f"/epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), ckpt_dir)

        # Early stopping: check if lr is < 1e-6
        if scheduler is not None:
            if optimizer.param_groups[0]["lr"] < 1e-6:
                print(f"Early stopping at epoch {epoch}: lr <= 1e-6")
                break

    # Test and save last model only if it was not saved above
    if (epoch+1) % args.test_every != 0:
        validate_one_epoch(model, test_loader, device, epoch, writer, "test", args)
        ckpt_dir = writer.log_dir + f"/epoch_{epoch}.ckpt"
        torch.save(model.state_dict(), ckpt_dir)

    # Close Tensorboard writer
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training Script')

    # Training
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloaders')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'vit'],
                        help='Model architecture (resnet or vit)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lrscheduler', action='store_true', help='Use LinearLR scheduler.')
    parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (in epochs)')
    parser.add_argument('--test_every', type=int, default=5, help='Testing frequency (in epochs)')
    # Experiment
    parser.add_argument('--seed', type=int, help='Seed for reproducing the experiment results.')
    parser.add_argument('--validation_step_before_train', action='store_true',
                        help='Executes validation before training.')
    parser.add_argument('--split_n', type=int, default=None, help='Number of split to use for the experiment.')
    # Log
    parser.add_argument('--log_dir', type=str, default='controlled_experiment/base_models/models',
                        help='Directory for Tensorboard logs, violations and ckpt.')
    parser.add_argument('--id', type=str, default='vanilla', help='Unique experiment identifier')
    parser.add_argument('--reset_version_dir', action='store_true', default=False,
                        help='Resets the version directory.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to model ckpt.')
    # Properties
    parser.add_argument('--lambda_mult', type=float, default=1e-1, help='Property lambda multiplier.')
    parser.add_argument('--properties', type=str, nargs='+', default=[], help='List of properties to check')
    parser.add_argument('--lambda_pl', type=float, default=1, help='Property loss multiplier.')
    parser.add_argument('--prop_loss', action='store_true', help='Enable property loss')
    parser.add_argument('--new_prop_loss', action='store_true', help='Enable new property loss')
    parser.add_argument('--mask_loss', action='store_true',
                        help='Enable masking main loss based on property violations')
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')

    args = parser.parse_args()
    main(args)
