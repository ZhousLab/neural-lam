import os
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
import yaml

from neural_lam.tcwv_dataset import TCWVDataset
from neural_lam import constants

def main():
    parser = ArgumentParser(description='Training arguments')
    parser.add_argument('--dataset', type=str, default="tcwv05",
        help='Dataset to compute weights for (default: meps_example)')
    parser.add_argument('--batch_size', type=int, default=4,
        help='Batch size when iterating over the dataset')
    parser.add_argument('--step_length', type=int, default=1,
        help='Step length in hours to consider single time step (default: 3)')
    parser.add_argument('--n_workers', type=int, default=0,
        help='Number of workers in data loader (default: 1)')
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # === load config ===
    config_path = '/root/Desktop/machine_learning/neural-lam/data/tcwv05/task.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    split = 'train'
    n_t = config[split]['n_t']
    pred_length = config['task']['pred_length']

    # Create parameter weights based on height
    # based on fig A.1 in graph cast paper
    # w_par = np.zeros((len(constants.param_names),))
    # w_dict = {'2': 1.0, '0': 0.1, '65': 0.065, '1000': 0.1, '850': 0.05, '500': 0.03}
    # w_list = np.array([w_dict[par.split('_')[-2]] for par in constants.param_names])
    w_list = np.ones(1)
    print("Saving parameter weights...")
    np.save(os.path.join(static_dir_path, 'parameter_weights.npy'),
            w_list.astype('float32'))

    # Load dataset without any subsampling: subsample_step=1
    ds = TCWVDataset(args.dataset, split=split, subsample_step=1, pred_length=pred_length,
            standardize=False) # Without standardization
    loader = torch.utils.data.DataLoader(ds, args.batch_size, shuffle=False,
            num_workers=args.n_workers)
    # Compute mean and std.-dev. of each parameter (+ flux forcing) across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, _, forcing_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch),
                dim=1) # (N_batch, N_t, N_grid, d_features)
        batch = init_batch
        means.append(torch.mean(batch, dim=(1,2))) # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1,2))) # (N_batch, d_features,)

        flux_batch = forcing_batch[:,:,:,0] # Flux is first index
        flux_means.append(torch.mean(flux_batch)) # (,)
        flux_squares.append(torch.mean(flux_batch**2)) # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0) # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2) # (d_features)

    flux_mean = torch.mean(torch.stack(flux_means)) # (,)
    flux_second_moment = torch.mean(torch.stack(flux_squares)) # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2) # (,)
    flux_stats = torch.stack((flux_mean, flux_std))

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))
    torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = TCWVDataset(args.dataset, split="train", subsample_step=1,
            pred_length=pred_length, standardize=True) # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(ds_standard, args.batch_size,
            shuffle=False, num_workers=args.n_workers)
    used_subsample_len = (n_t//args.step_length)*args.step_length

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _ in tqdm(loader_standard):
        batch = torch.cat((init_batch, target_batch),
                dim=1) # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1h-steps
        # stepped_batch = torch.cat([batch[:,ss_i:used_subsample_len:args.step_length]
        #     for ss_i in range(args.step_length)], dim=0)
        # (N_batch', N_t, N_grid, d_features), N_batch' = args.step_length*N_batch

        # batch_diffs = stepped_batch[:,1:] - stepped_batch[:,:-1]
        batch_diffs = batch[:,1:] - batch[:,:-1]
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(torch.mean(batch_diffs, dim=(1,2))) # (N_batch', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2,
            dim=(1,2))) # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0) # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2) # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))

if __name__ == "__main__":
    main()
