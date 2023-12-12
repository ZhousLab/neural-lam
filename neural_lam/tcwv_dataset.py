import os
import glob
import torch
import numpy as np
import datetime as dt
import yaml

from neural_lam import utils, constants

class TCWVDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t' = 20826
    N_t = 20826//subsample_step 
    d_features = 1 (tcwv)
    d_forcing = 2 (sin + cos)
    """
    def __init__(self, dataset_name, pred_length=5, split="train", subsample_step=3,
            standardize=True, subset=False):
        super().__init__()
        # === load task config ===
        config_path = '/root/Desktop/machine_learning/neural-lam/data/tcwv05/task.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.n_t = config[split]['n_t']
        # === init WeatherDataset ===
        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)
        self.standardize = standardize
        self.file_name = 'tcwv.npy'  # we only have one file
        if subset:
            self.sample_names = self.sample_names[:50] # Limit to 50 timesteps, for testing
        self.sample_length = pred_length + 2 # 2 init states, total window length when slicing data
        self.subsample_step = subsample_step     # not sure the purpose of this
        # self.n_samples = 100
        self.n_samples = self.n_t//self.sample_length # number of samples in dataset

        # Set up for z score standardization
        self.standardize = standardize    # skip for now
        if standardize:
            # pass
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std, self.flux_mean, self.flux_std =\
                ds_stats["data_mean"], ds_stats["data_std"], ds_stats["flux_mean"], \
                ds_stats["flux_std"]

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"    # not sure the purpose of this

        # === Load sample (tcwv) ===
        sample_path = os.path.join(self.sample_dir_path, self.file_name)
        # print cwd
        # print(os.getcwd())
        # print sample_path
        # print(sample_path)
        # sample_path = '/root/Desktop/machine_learning/neural-lam/data/tcwv0.5/samples/train/tcwv.npy'
        # # (N_t, N_x, N_y, d_features')
        sample = torch.tensor(np.load(sample_path), dtype=torch.float32)
        # sample = torch.tensor(np.load('/root/Desktop/machine_learning/neural-lam/data/tcwv0.5/samples/train/tcwv.npy'))
        # Flatten spatial dim
        self.sample = sample.flatten(1,2) # (N_t, N_grid, d_features)
        # print(f'sample shape: {sample.shape}')

        # === Load static data ===
        lan_sea_mask_path = os.path.join(self.sample_dir_path, 'land_sea_mask.npy')
        land_sea_mask = torch.tensor(np.load(lan_sea_mask_path), dtype=torch.float32)  
        self.land_sea_mask = land_sea_mask.flatten(0) # (N_grid,)
        self.land_sea_mask = self.land_sea_mask[:,None] # (N_grid, 1)
        # pass for now

        # === Forcing features ===
        # # Forcing features - time of year embedding
        sin_feature_path = os.path.join(self.sample_dir_path, 'sin_feature.npy')
        cos_feature_path = os.path.join(self.sample_dir_path, 'cos_feature.npy')
        sin_features = torch.tensor(np.load(sin_feature_path),
                dtype=torch.float32)
        cos_features = torch.tensor(np.load(cos_feature_path),
                dtype=torch.float32)
        # sin_features = torch.tensor(np.load('/root/Desktop/machine_learning/neural-lam/data/tcwv0.5/samples/train/sin_feature.npy'),
        #         dtype=torch.float32)
        # cos_features = torch.tensor(np.load('/root/Desktop/machine_learning/neural-lam/data/tcwv0.5/samples/train/cos_feature.npy'),
        #         dtype=torch.float32)
        # flatten
        sin_features = sin_features.flatten(1,2)
        cos_features = cos_features.flatten(1,2)
        self.forcing_features = torch.stack((sin_features, cos_features), dim=2)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # === Sample ===
        sample = self.sample[idx*self.sample_length:(idx+1)*self.sample_length]
        # (sample_length, N_grid, d_features)

        # Split up sample in init. states and target states
        init_states = sample[:2,...] # (2, N_grid, d_features)
        target_states = sample[2:,...] # (sample_length-2, N_grid, d_features)

        # === Static batch features ===
        static_features = self.land_sea_mask

        # === Forcing ===
        forcing = self.forcing_features[idx*self.sample_length:(idx+1)*self.sample_length]
        # init_forcing = forcing[:2,...]
        target_forcing = forcing[2:,...]
        # stack init_states and init_forcing
        # init_states = torch.cat((init_states, init_forcing), dim=2) # (2, N_grid, d_features+d_forcing)

        return init_states, target_states, static_features, target_forcing