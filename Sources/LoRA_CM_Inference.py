import os
import json
import torch
import numpy as np
from tqdm import tqdm
from model_lora import build_lora_unet
from model import OneDUnet
import pandas as pd

def computeKi(K1, k2, k3):
    return (K1 * k3)/(k2 + k3)

def Sampler(model, y, num_posterior_samples, x_dim, t_list=[1.0, 0.75, 0.50, 0.25]):
    """
    model: either DataParallel or single-GPU model
    y:      Tensor of shape [B, y_dim]
    x_dim:  Dimension of the output (target) variable
    returns: Tensor of shape [B, num_posterior_samples, x_dim]
    """
    batch_size, y_dim = y.shape
    # Ensure t_list is on the correct device
    device = y.device
    t_list = torch.tensor(t_list, dtype=torch.float32, device=device)

    # repeat y
    y_rep = y.unsqueeze(1).repeat(1, num_posterior_samples, 1)
    y_rep = y_rep.view(-1, y_dim)  # [B * S, y_dim]

    # initialize noise
    x_noisy = torch.randn(batch_size * num_posterior_samples, 1, x_dim, device=device)

    for idx, tau in enumerate(t_list):
        # scalar tensor
        t_tensor = torch.full((batch_size * num_posterior_samples,), tau, device=device)

        if idx > 0:
            z = torch.randn_like(x_noisy, device=device)
            x_noisy = x_noisy + tau * z

        with torch.no_grad():
            x_noisy = model(x_noisy, y_rep, t_tensor)

    # reshape to [B, S, x_dim]
    x_noisy = x_noisy.view(batch_size, num_posterior_samples, x_dim)
    return x_noisy

def Load_Model(ckpt_path, model_params_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # ---------- initiate a new model ----------
    with open(model_params_path, "r") as f:
        cfg = json.load(f)
    
    try:
        # Build model using saved config dimensions
        lora_model = build_lora_unet(
            x_dim=cfg["x_dim"],
            y_dim=cfg["y_dim"],
            embed_dim=cfg["embed_dim"],
            channels=cfg["channels"],
            embedy=cfg["embedy"],
            sigma_data=cfg["sigma_data"],
            r=cfg["lora_r"],
            alpha=cfg["lora_alpha"],
            dropout=cfg["lora_dropout"],
            pretrained_ckpt=None, # Important: We load specific weights below, don't re-init from foundation here
            train_norms=False,
            device=device,
        )

        # ---------- load weights ----------
        state_dict = ckpt["model_state_dict"]

        # Strict=True is fine here because the model we built exactly matches the saved config
        lora_model.load_state_dict(state_dict, strict=True)
        lora_model.to(device)
        lora_model.eval()
    
    except: # if LoRA-CM is not used, go back to the pretrained CM
        lora_model = OneDUnet(x_dim=cfg["x_dim"], 
                         y_dim=cfg["y_dim"], 
                         embed_dim=cfg["embed_dim"], 
                         channels=cfg["channels"], 
                         embedy=cfg["embedy"], 
                         sigma_data=cfg["sigma_data"]).to(device)
        state_dict = ckpt["model_state_dict"]
        lora_model.load_state_dict(state_dict, strict=True)
        lora_model.to(device)
        lora_model.eval()

    return lora_model, cfg

def inference(data_path, ROI_voxel_idx_path, model_path, output_path, device, batch_size, sample_size, save_full_posterior=False):
    # ------ CM Prediction Hyperparameters ------
    num_timesteps   = 3 # if s=15 use 3 or 5 if s=50 use 2 or 5
    t_list          = np.linspace(1.0, 0.0, num=num_timesteps, endpoint=False)

    # ------ Load dynamic PET data ------
    y_data = pd.read_hdf(data_path)
    AIF    = y_data.iloc[:,2].values # Arterial input function column index 2
    y_mat  = y_data.iloc[:,3:].to_numpy().T  # TACs start from column index 3
    AIFs   = np.tile(AIF, (y_mat.shape[0], 1))
    y_full = np.hstack((y_mat, AIFs))

    # ------ Load segmentation data ------
    ROI_voxel_idx = np.load(ROI_voxel_idx_path)
    ROIs = list(ROI_voxel_idx)

    # ------ Pre-fetch x_dim and Initialize Array ------
    first_organ = ROIs[0]
    wd = model_path[first_organ][0]
    model_params_path = "{}/model_params.json".format(wd)
    with open(model_params_path, "r") as f:
        cfg = json.load(f)
    x_dim = cfg["x_dim"]

    if save_full_posterior:
        # Shape: [Voxels, Sample_Size, x_dim]
        final_estimates = np.zeros((y_full.shape[0], sample_size, x_dim), dtype=np.float32)
    else:
        # Shape: [Voxels, 1] for Ki mean
        final_estimates = np.zeros((y_full.shape[0], 1), dtype=np.float32)

    # ------ Inference starts here ------
    for organ in ROIs:
        print(f'Now processing {organ} voxels...')

        wd = model_path[organ][0]
        ckpt_path = "{}/{}".format(wd, model_path[organ][1])
        scaling_path = "{}/scaling_params.json".format(wd)
        model_params_path = "{}/model_params.json".format(wd)
        lora_model, cfg = Load_Model(ckpt_path, model_params_path, device)
        # Extract x_dim from the config we just loaded
        x_dim = cfg["x_dim"]

        ### Get data normalisation parameters ready
        with open(scaling_path, "r") as f:
            scaling = json.load(f)
            x_mean = torch.tensor(np.array(scaling["x_mean"][:x_dim]), dtype=torch.float32).to(device)
            x_std  = torch.tensor(np.array(scaling["x_std"][:x_dim]), dtype=torch.float32).to(device)
        y_mean = np.array(scaling["y_mean"])
        y_std  = np.array(scaling["y_std"])

        ### Get data ready
        indices = ROI_voxel_idx[f'{organ}']
        y = y_full[indices, :]
        y_norm = (y - y_mean) / y_std

        ### Start prediction
        num_rows = y_norm.shape[0]
        outputs  = []
        for i in tqdm(range(0, num_rows, batch_size)):
            batch_np = y_norm[i:i+batch_size]
            obs      = torch.tensor(batch_np, dtype=torch.float32, device=device)

            # ----- multi-GPU inference -----
            theta_hat_cpu = Sampler(lora_model, obs, sample_size, x_dim, t_list)

            # bring to GPU0 to denormalize
            theta_hat = theta_hat_cpu.to(device)

            # denormalize / exp
            x_pred = theta_hat * x_std + x_mean
            samples = x_pred.cpu().numpy()

            # prepare to save
            batch_indices = indices[i:i+batch_size]

            if save_full_posterior:
                # Slot the 3D block directly into the big array
                final_estimates[batch_indices] = samples
            else:
                # Calculate Ki mean and slot the 1D block directly in
                Ki = computeKi(samples[:,:,0], samples[:,:,1], samples[:,:,2])
                Ki_mean = Ki.mean(axis=1)
                final_estimates[batch_indices] = Ki_mean.reshape(-1, 1)
            
        print(f'✅ {organ} is processed!!!')
    
    # Save & Create the parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, arr = final_estimates)