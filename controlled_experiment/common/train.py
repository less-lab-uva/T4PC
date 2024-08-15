import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
import json
import shutil
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from controlled_experiment.common.resnet_model import ResnetModel
from controlled_experiment.common.data import CARLA_Data
from controlled_experiment.common.tcp_config import GlobalConfig

def init_summary_writer(log_dir, reset_version_dir=False):
    """
    Initialize a Tensorboard SummaryWriter with the appropriate version number folder.

    Parameters:
    - log_dir (str): The base directory for Tensorboard logs.
    - reset_version_dir (bool): Whether to reset the version directory.

    Returns:
    - writer (SummaryWriter): Tensorboard SummaryWriter.
    """
    # Create the log_dir if it does not exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Scan existing version folders
    existing_versions = [d for d in os.listdir(log_dir) if d.startswith('version_') and os.path.isdir(os.path.join(log_dir, d))]

    # Determine the next version number
    if not existing_versions:
        version_number = 0
    else:
        if reset_version_dir:
            version_number = 0
            if (Path(log_dir) / "version_0").exists():
                shutil.rmtree(os.path.join(log_dir, "version_0"))
        else:
            version_numbers = [int(d.split('_')[1]) for d in existing_versions]
            version_number = max(version_numbers) + 1

    # Create the next version folder
    next_version_folder = os.path.join(log_dir, f'version_{version_number}')
    os.makedirs(next_version_folder, exist_ok=True)

    # Initialize Tensorboard SummaryWriter
    writer = SummaryWriter(log_dir=next_version_folder)

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
		return torch.mean((prop_reference_output - prop_reference_output)), mask.count_nonzero().item(), mask
	if loss_type == "mse":
		return torch.mean((pred[mask] - prop_reference_output[mask])**2), mask.count_nonzero().item(), mask
	elif loss_type == "mae":
		return torch.mean(torch.abs(pred[mask] - prop_reference_output[mask])), mask.count_nonzero().item(), mask
	else:
		raise NotImplementedError("Loss type not implemented.")

def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer, scaler, lr_mult, args):
    model.train()
    total_loss = 0.0
    total_steering_loss = 0.0
    total_acceleration_loss = 0.0
    prop_loss_dict = {}
    prop_count_dict = {}
    for prop_name in args.properties:
        prop_loss_dict[f"train_{prop_name}_loss"] = 0.0
        prop_count_dict[f"train_violation_count_{prop_name}"] = 0.0

    # Wrap the train_loader with tqdm for the progress bar
    tqdm_train_loader = tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=True)
    violation_list = {}

    for batch_idx, batch in enumerate(tqdm_train_loader):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            step = epoch * len(train_loader) + batch_idx

            inputs = batch['front_img'].to(device)
            acceleration = batch['acceleration'].to(device)
            steering_angle = batch['steer'].to(device)

            optimizer.zero_grad()
            # optimizer_prop_loss.zero_grad()
            # [throttle, steer, brake, acceleration]
            outputs = model(inputs)

            acc_prop_mask = torch.ones(batch['acceleration'].shape[0], device=device)
            steer_prop_mask = torch.ones(batch['steer'].shape[0], device=device)

            # Property loss
            all_prop_loss = []
            for i in range(len(batch["prop_eval"])):
                prop_name = batch["prop_name"][i][0]
                if prop_name in args.properties:
                    prop_attribute = "acceleration_continuous" if batch["prop_attribute"][i][0] == "acceleration" else "steering_angle"

                    # Mask v1: use wrong labels to mask main loss
                    if prop_attribute == "acceleration_continuous":
                        _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), acceleration, batch["prop_comparison"][i][0])
                        acc_prop_mask *= (~mask).long()
                    elif prop_attribute == "steering_angle":
                        _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), steering_angle, batch["prop_comparison"][i][0])
                        steer_prop_mask *= (~mask).long()
                    else:
                        raise NotImplementedError("No mask for property attribute")
                    
                    prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), outputs[prop_attribute], batch["prop_comparison"][i][0])
                    all_prop_loss.append((prop_name, prop_loss * args.lambda_pl))
                    prop_loss_dict[f"train_{prop_name}_loss"] += prop_loss.item()
                    prop_count_dict[f"train_violation_count_{prop_name}"] += vc

            # Main Losses
            if args.mask_loss:
                acceleration_loss = F.mse_loss(outputs["acceleration_continuous"]*acc_prop_mask, acceleration*acc_prop_mask)
                if acc_prop_mask.count_nonzero().item() > 0:
                    acceleration_loss *= len(acc_prop_mask) / acc_prop_mask.count_nonzero().item()
                steering_angle_loss = F.mse_loss(outputs["steering_angle"]*steer_prop_mask, steering_angle*steer_prop_mask)
                if steer_prop_mask.count_nonzero().item() > 0:
                    steering_angle_loss *= len(steer_prop_mask) / steer_prop_mask.count_nonzero().item()
            else:
                acceleration_loss = F.mse_loss(outputs["acceleration_continuous"], acceleration)
                steering_angle_loss = F.mse_loss(outputs["steering_angle"], steering_angle)
            total_acceleration_loss += acceleration_loss.item()
            total_steering_loss += steering_angle_loss.item()
            losses = acceleration_loss + steering_angle_loss

            for p_name, p_loss in all_prop_loss:
                # Apply property loss
                if args.prop_loss:
                    losses += lr_mult[p_name] * p_loss
                # Save losses for lambda update
                if p_name not in violation_list.keys():
                    violation_list[p_name] = [p_loss.item()]
                else:
                    violation_list[p_name].append(p_loss.item())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        # scaler.step(optimizer_prop_loss)
        scaler.update()
        optimizer.zero_grad()
        # optimizer_prop_loss.zero_grad()

        total_loss += losses.item()
    
    before_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar('Learning Rate', before_lr, epoch)
    scheduler.step()
    updated_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: lr %.2E -> %.2E" % (epoch, before_lr, updated_lr))

    for key in prop_loss_dict.keys():
        writer.add_scalar(key, prop_loss_dict[key] / len(train_loader), epoch)
    for key in prop_count_dict.keys():
        writer.add_scalar(key, prop_count_dict[key], epoch)
    writer.add_scalar('Train Steer Loss (Epoch)', total_steering_loss / len(train_loader), epoch)
    writer.add_scalar('Train Acceleration Loss (Epoch)', total_acceleration_loss / len(train_loader), epoch)
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Train Loss (Epoch)', avg_loss, epoch)

    print(lr_mult)
    for p_name in lr_mult.keys():
        lr_mult[p_name] += args.lambda_mult * np.mean(violation_list[p_name])
        writer.add_scalar(f'{p_name} loss lambda after (Epoch)', lr_mult[p_name], epoch)
    print(lr_mult)
    
    return avg_loss, lr_mult

def validate_one_epoch(model, val_loader, device, epoch, writer, scaler, args):
    model.eval()
    total_loss = 0.0
    total_loss_no_pl = 0.0
    total_steering_loss = 0.0
    total_acceleration_loss = 0.0
    prop_loss_dict = {}
    prop_count_dict = {}
    for prop_name in args.properties:
        prop_loss_dict[f"val_{prop_name}_loss"] = 0.0
        prop_count_dict[f"val_violation_count_{prop_name}"] = 0.0

    # Wrap the val_loader with tqdm for the progress bar
    tqdm_val_loader = tqdm(val_loader, desc=f'Val Epoch {epoch}', leave=False)
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

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_val_loader):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                step = epoch * len(val_loader) + batch_idx
                inputs = batch['front_img'].to(device)
                acceleration = batch['acceleration'].to(device)
                steering_angle = batch['steer'].to(device)

                # [acceleration, steer, brake, acceleration_continuous]
                outputs = model(inputs)

                acc_prop_mask = torch.ones(batch['acceleration'].shape[0], device=device)
                steer_prop_mask = torch.ones(batch['steer'].shape[0], device=device)

                # Property loss
                all_prop_loss = []
                for i in range(len(batch["prop_eval"])):
                    prop_name = batch["prop_name"][i][0]
                    if prop_name in args.properties:
                        prop_attribute = "acceleration_continuous" if batch["prop_attribute"][i][0] == "acceleration" else "steering_angle"
                        
                        # Mask v1: use wrong labels to mask main loss
                        if prop_attribute == "acceleration_continuous":
                            _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), acceleration, batch["prop_comparison"][i][0])
                            acc_prop_mask *= (~mask).long()
                        elif prop_attribute == "steering_angle":
                            _,_,mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), steering_angle, batch["prop_comparison"][i][0])
                            steer_prop_mask *= (~mask).long()
                        else:
                            raise NotImplementedError("No mask for property attribute")

                        prop_loss, vc, mask = my_own_loss(batch["prop_eval"][i].to(device), batch["prop_reference_output"][i].to(device), outputs[prop_attribute], batch["prop_comparison"][i][0])
                        if args.prop_loss:
                            all_prop_loss.append(prop_loss * args.lambda_pl)
                        prop_loss_dict[f"val_{prop_name}_loss"] += prop_loss.item()
                        prop_count_dict[f"val_violation_count_{prop_name}"] += vc

                        data["violation"].extend(mask.cpu().tolist())
                        data["img_id"].extend(list(batch['img_dir'][0]))
                        data["acc_label"].extend(batch['acceleration'].tolist())
                        data["acc_prop_mask"].extend(acc_prop_mask.tolist())
                        data["steer_label"].extend(batch['steer'].tolist())
                        data["steer_prop_mask"].extend(steer_prop_mask.tolist())
                        data["prop_name"].extend([prop_name] * len(mask))
                        data["acc_pred"].extend(outputs["acceleration_continuous"].cpu().tolist())
                        data["steer_pred"].extend(outputs["steering_angle"].cpu().tolist())

                # Main Losses
                if args.mask_loss:
                    acceleration_loss = F.mse_loss(outputs["acceleration_continuous"]*acc_prop_mask, acceleration*acc_prop_mask)
                    if acc_prop_mask.count_nonzero().item() > 0:
                        acceleration_loss *= len(acc_prop_mask) / acc_prop_mask.count_nonzero().item()
                    steering_angle_loss = F.mse_loss(outputs["steering_angle"]*steer_prop_mask, steering_angle*steer_prop_mask)
                    if steer_prop_mask.count_nonzero().item() > 0:
                        steering_angle_loss *= len(steer_prop_mask) / steer_prop_mask.count_nonzero().item()
                else:
                    acceleration_loss = F.mse_loss(outputs["acceleration_continuous"], acceleration)
                    steering_angle_loss = F.mse_loss(outputs["steering_angle"], steering_angle)
                total_acceleration_loss += acceleration_loss.item()
                total_steering_loss += steering_angle_loss.item()
                losses = acceleration_loss + steering_angle_loss
                total_loss_no_pl += losses.item()

                if args.prop_loss:
                    for p_loss in all_prop_loss:
                        losses += p_loss

            scaler.scale(losses)
            total_loss += losses.item()

    r = prop_loss_dict.copy()
    if writer is not None:
        for key in prop_loss_dict.keys():
            writer.add_scalar(key, prop_loss_dict[key] / len(val_loader), epoch)
            r[key] = prop_loss_dict[key] / len(val_loader)
        for key in prop_count_dict.keys():
            writer.add_scalar(key, prop_count_dict[key], epoch)
        writer.add_scalar('Validation Steer Loss (Epoch)', total_steering_loss / len(val_loader), epoch)
        writer.add_scalar('Validation Acceleration Loss (Epoch)', total_acceleration_loss / len(val_loader), epoch)
        writer.add_scalar('Validation Loss No PL (Epoch)', total_loss_no_pl / len(val_loader), epoch)
        writer.add_scalar('Validation Loss (Epoch)', total_loss / len(val_loader), epoch)

    r["val_avg_steering_loss"] = total_steering_loss / len(val_loader)
    r["val_avg_acceleration_loss"] = total_acceleration_loss / len(val_loader)
    r["val_avg_total_loss_no_pl"] = total_loss_no_pl / len(val_loader)
    r["val_avg_total_loss"] = total_loss / len(val_loader)

    return r, pd.DataFrame(data)

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed used: {args.seed}")
    else:
        torch.manual_seed(0)
        print(f"Seed used: 0")

    args.log_dir = os.path.join(args.log_dir, args.id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train_sampling_table is not None:
        train_sampling_table = pd.read_csv(args.train_sampling_table)
    else:
        train_sampling_table = None
    if args.val_sampling_table is not None:
        val_sampling_table = pd.read_csv(args.val_sampling_table)
    else:
        val_sampling_table = None

    # Initialize your dataset and dataloaders
    config = GlobalConfig(split=args.split_n)
    train_dataset = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug, transform_for=args.model_type, sampling_table=train_sampling_table, data_id="packed_data_ce.npy")
    print(f"Length of train: {len(train_dataset)}")
    val_dataset = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data, transform_for=args.model_type, sampling_table=val_sampling_table, data_id="packed_data_ce.npy")
    print(f"Length of val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

    # Initialize your model
    model = ResnetModel()

    # Load weights
    if args.model_ckpt is not None:
        model.load_state_dict(torch.load(args.model_ckpt, map_location=torch.device('cpu')))

    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=0)

    # Tensorboard writer
    writer = init_summary_writer(args.log_dir, args.reset_version_dir)
    logdir_path = Path(args.log_dir)
    model_name = logdir_path.name
    base_path = logdir_path / "../../"
    if not (base_path/"violations").exists():
        (base_path/"violations").mkdir(parents=True, exist_ok=True)

    # Training loop
    if args.validation_step_before_train:
        val_loss, data_df = validate_one_epoch(model, val_loader, device, 0, writer, scaler, args)
        # data_df.to_csv(base_path/f"violations/val__e0__{model_name}.csv", index=False)
    best_val_loss = float('inf')  # Initialize with a large value
    lr_mult = {}
    for prop in args.properties:
        lr_mult[prop] = 0
    for epoch in range(0, args.epochs):
        train_loss, lr_mult = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, writer, scaler, lr_mult, args)

        if (epoch+1) % args.val_every == 0:
            # Val
            val_loss_dict, data_df = validate_one_epoch(model, val_loader, device, epoch+1, writer, scaler, args)
            # data_df.to_csv(base_path/f"violations/val__e{epoch+1}__{model_name}.csv", index=False)

            with open(os.path.join(writer.log_dir, "val_loss.json"), "w") as f:
                json.dump(val_loss_dict, f)
            data_df.to_csv(os.path.join(writer.log_dir, "val_data.csv"), index=False)
        
        if args.model_versioning:
            ckpt_dir = writer.log_dir + f"/epoch_{epoch}.ckpt"
            torch.save(model.state_dict(), ckpt_dir)
        
        ckpt_dir = writer.log_dir + "/last_model.ckpt"
        torch.save(model.state_dict(), ckpt_dir)

    # Close Tensorboard writer
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_every', type=int, default=1, help='Validation frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--log_dir', type=str, default='controlled_experiment/base_models/models', help='Directory for Tensorboard logs and ckpt.')
    parser.add_argument('--id', type=str, default='vanilla', help='Unique experiment identifier')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloaders')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'vit'], help='Model architecture (resnet or vit)')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to model ckpt.')
    parser.add_argument('--model_versioning', action='store_true', help='Save different versions of the model when it improves')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training (fp16)')
    parser.add_argument('--properties', type=str, nargs='+', default=[], help='List of properties to check')
    parser.add_argument('--lambda_pl', type=float, default=1.0, help='Property loss multiplier.')
    parser.add_argument('--prop_loss', action='store_true', help='Enable property loss')
    parser.add_argument('--mask_loss', action='store_true', help='Enable masking main loss based on property violations')
    parser.add_argument('--validation_step_before_train', action='store_true', help='Enable property loss')
    parser.add_argument('--reset_version_dir', action='store_true', default=False, help='Enable property loss')
    parser.add_argument('--train_sampling_table', type=str, help='Path to train csv sampling table.')
    parser.add_argument('--val_sampling_table', type=str, help='Path to val csv sampling table.')
    parser.add_argument('--seed', type=int, help='Seed for reproducing the experiment results.')
    parser.add_argument('--split_n', type=int, default=None, help='Number of split to use for the experiment.')
    parser.add_argument('--lambda_mult', type=float, default=1e-3, help='Property lambda multiplier.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')
    args = parser.parse_args()
    main(args)
