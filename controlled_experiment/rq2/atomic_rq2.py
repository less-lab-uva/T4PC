import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from controlled_experiment.common.tcp_config import GlobalConfig
from controlled_experiment.common.data import CARLA_Data
from controlled_experiment.common.resnet_model import ResnetModel
from controlled_experiment.common.utils import get_val_violations
from types import SimpleNamespace
from pathlib import Path
from controlled_experiment.common.train import main as vanilla_train
from controlled_experiment.common.train import my_own_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", help="Path to model's checkpoint.")
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'vit'], help='Model architecture.')
    parser.add_argument('--train_sampling_table', type=str, help='Path to train csv sampling table.')
    parser.add_argument('--val_sampling_table', type=str, help='Path to val csv sampling table.')
    parser.add_argument('--sampling_per', type=int, help='Sampling percentage to be used.')
    parser.add_argument('--properties', type=str, nargs='+', help='List of properties to check')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--run_n', type=int, default=0, help='Number of run for the experiment.')
    parser.add_argument('--split_n', type=int, default=None, help='Number of split to use for the experiment.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducing the experiment results.')
    parser.add_argument('--prop_loss', action='store_true', help='Enable property loss')
    parser.add_argument('--mask_loss', action='store_true', help='Enable masking main loss based on property violations')
    parser.add_argument('--use_regularization_images', action='store_true', default=True, help='Wheter to use regularization images or not.')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'], help='Optimizer to be used.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')
    args = parser.parse_args()

    # assert len(args.properties) == 1, "Only one property must be provided."
    model_id = None
    if len(args.properties) > 1:
        # Only base model is allowed
        assert len(args.properties) == 6, "Only one property or all must be provided."
        assert not args.prop_loss, "Base model should not have property loss activated."
        assert not args.mask_loss, "Base model should not have mask loss activated."
        model_id = f"base__split{args.split_n}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Partition dataset
    base_path = Path(f"controlled_experiment/rq2/data/{args.model_type}/") 

    # Create arguments for vanilla training
    vanilla_args_dict = {
        "model_ckpt": args.model_ckpt,
        "model_type": args.model_type,
        "prop_loss": args.prop_loss,
        "mask_loss": args.mask_loss,
        "validation_step_before_train": True,
        "reset_version_dir": False,
        "lambda_pl": 1,
        "properties": args.properties,
        "log_dir": base_path/"models",
        "id": f"{args.properties[0]}__{'pl' if args.prop_loss else 'nopl'}__split{args.split_n}" if model_id is None else model_id,
        "train_sampling_table": None,
        "val_sampling_table": None,
        "epochs": 15,
        "batch_size": args.batch_size,
        "val_every": 1,
        "num_workers": 8,
        "lr": 1e-5,
        "model_versioning": False,
        "mixed_precision": False,
        "lrscheduler": True,
        "seed": args.seed,
        "split_n": args.split_n,
        "lambda_mult": 1e-1,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay
    }
    vanilla_args = SimpleNamespace(**vanilla_args_dict)

    # Call vanilla training script
    vanilla_train(vanilla_args)

    # Initialize your model
    model = ResnetModel()

    # Load weights
    aux_model_path = f"{args.properties[0]}__{'pl' if args.prop_loss else 'nopl'}__split{args.split_n}"
    aux_model_path = f"{args.properties[0]}__{'pl' if args.prop_loss else 'nopl'}__split{args.split_n}" if model_id is None else model_id
    model.load_state_dict(torch.load(base_path/f"models/{aux_model_path}/version_0/last_model.ckpt", map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    
    # Do a pass on dataset to get the number of violations after using property loss
    for partition in ["test"]:
        config = GlobalConfig(split=args.split_n)
        partition_config = config.train_data if partition == "train" else config.val_data if partition == "val" else config.test_data
        dataset = CARLA_Data(root=config.root_dir_all, data_folders=partition_config, img_aug=config.img_aug, transform_for=vanilla_args.model_type, data_id="packed_data_ce.npy")
        loader = DataLoader(dataset, batch_size=vanilla_args.batch_size, shuffle=False, num_workers=vanilla_args.num_workers, persistent_workers=True)
        tqdm_loader = tqdm(loader, desc=f'Calculating number of violations', leave=False)
        data = get_val_violations(tqdm_loader, device, model, my_own_loss, vanilla_args)

        # Create df from data and save it
        df = pd.DataFrame(data)
        if not (base_path/"violations").exists():
            (base_path/"violations").mkdir(parents=True, exist_ok=True)
        df.to_csv(base_path/f"violations/{partition}__{aux_model_path}.csv", index=False)
    
if __name__ == "__main__":
    main()