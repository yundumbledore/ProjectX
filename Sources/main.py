from LoRA_CM_Inference import inference
from Visualisation import parametric_imaging

def run_inference(data_path, ROI_voxel_idx_path, output_path, save_full_posterior, device):
    ## ------ Define LoRA-CM Models ------
    # Example: Region-specific prediction needs to define model paths for each ROI
    # "model_path" dictionary keys are a two-element array: 1st is the model directory, 2nd is the best model
    model_path = {"Brain": ["./LoRA_CM_Models/FDG-2TCM-brain_LoRA_lr0.0001_r4_alpha8", "valbest_epoch0136_trainloss0.00126482_valloss0.00097434.pth"],
                "Kidneys": ["./LoRA_CM_Models/FDG-2TCM-kidneys_LoRA_lr0.0001_r4_alpha8", "valbest_epoch0103_trainloss0.00148140_valloss0.00124347.pth"],
                "Liver": ["./LoRA_CM_Models/FDG-2TCM-liver_LoRA_lr0.0001_r4_alpha8", "valbest_epoch0181_trainloss0.00148765_valloss0.00115688.pth"],
                "Lungs": ["./LoRA_CM_Models/FDG-2TCM-lungs_LoRA_lr0.0001_r4_alpha8", "valbest_epoch0189_trainloss0.00180177_valloss0.00144622.pth"],
                "Myocardium": ["./LoRA_CM_Models/FDG-2TCM-myocardium_LoRA_lr0.0001_r4_alpha8", "epoch0200_trainloss0.00149256_valloss0.00131118.pth"],
                "Miscellaneous": ["./LoRA_CM_Models/FDG-2TCM-miscellaneous_LoRA_lr0.0001_r4_alpha8", "valbest_epoch0169_trainloss0.00184525_valloss0.00156166.pth"],
                "Bladder": ["./Pretrained_CM", "pretrained.pth"]
                }

    ## ------ Define LoRA-CM Sampling Details ------
    batch_size = 1000
    sample_size = 1000

    ## ------ Estimate Parameters ------
    inference(data_path, ROI_voxel_idx_path, model_path, output_path, device, batch_size, sample_size, save_full_posterior)

def run_imaging(slice_index, mask_path, estimates_path, parameter_name, save_path):
    parametric_imaging(slice_index, mask_path, estimates_path, parameter_name, save_path)



