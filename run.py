import sys
import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings; warnings.filterwarnings("ignore")

dir_path = os.path.join(os.getcwd(), 'Sources')
sys.path.append(dir_path)

# Mock imports for demonstration; ensure these are available in your actual setup
from main import run_inference, run_imaging

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Primary device:", device)

### ------ 1. Settings ------

# If set to False, the GUI will be skipped entirely.
# If True, the GUI launches. If no display is found, it falls back to these settings.
USE_GUI = True 

Run_pipeline = ['Inference', 'Imaging']

# Inference Settings
data_path = './Example_Data/P001_PET_incl_bones_coronal_116_denoised_adjusted.h5'
ROI_voxel_idx_path = './Example_Data/P001_coronal_116_ROI_voxel_idx.npz'
estimates_output_path = './Results/P001_PET_incl_bones_coronal_116_Ki.npz'
save_full_posterior = False 

# Imaging Settings
estimates_path = './Results/P001_PET_incl_bones_coronal_116_Ki.npz'
mask_path = './Example_Data/P001_mask.npy'
imaging_output_path = './Results/P001_coronal_116_Ki.png'
parameter_name = 'Ki'
slice_index = 116

### ------ 2. Execution Logic ------

def execute_pipeline(pipeline, inf_data, inf_roi, inf_out, inf_save, 
                     img_est, img_mask, img_out, img_param, img_slice):
    """Runs the actual PyTorch tasks."""
    print(f"Starting execution on {device}...")
    
    # NOTE: Changed from if/elif to if/if so both can run sequentially!
    if 'Inference' in pipeline:
        print('\nNow Inference is running...')
        run_inference(inf_data, inf_roi, inf_out, inf_save, device)
        
    if 'Imaging' in pipeline:
        print('\nNow Imaging is running...')
        run_imaging(int(img_slice), img_mask, img_est, img_param, img_out)
        
    print("\nAll tasks completed!")

### ------ 3. GUI Definition ------

def launch_gui():
    """Builds and launches the Tkinter GUI."""
    root = tk.Tk()
    root.title("Dynamic PET Analysis Configuration")
    
    # Variables to hold GUI state, pre-filled with hardcoded defaults
    var_inf = tk.BooleanVar(value='Inference' in Run_pipeline)
    var_img = tk.BooleanVar(value='Imaging' in Run_pipeline)
    var_data = tk.StringVar(value=data_path)
    var_roi = tk.StringVar(value=ROI_voxel_idx_path)
    var_est_out = tk.StringVar(value=estimates_output_path)
    var_save_post = tk.BooleanVar(value=save_full_posterior)
    
    var_est_in = tk.StringVar(value=estimates_path)
    var_mask = tk.StringVar(value=mask_path)
    var_img_out = tk.StringVar(value=imaging_output_path)
    var_param = tk.StringVar(value=parameter_name)
    var_slice = tk.StringVar(value=str(slice_index))

    # Helper function to browse files
    def browse_file(var, is_save=False):
        if is_save:
            filepath = filedialog.asksaveasfilename()
        else:
            filepath = filedialog.askopenfilename()
        if filepath:
            var.set(filepath)

    # --- UI Layout ---
    row = 0
    
    # Pipeline Selection
    tk.Label(root, text="Tasks to Run:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(10,0), padx=10)
    row += 1
    tk.Checkbutton(root, text="Inference", variable=var_inf).grid(row=row, column=0, sticky="w", padx=20)
    tk.Checkbutton(root, text="Imaging", variable=var_img).grid(row=row, column=1, sticky="w")
    row += 1

    # Inference Section
    tk.Label(root, text="Inference Settings", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(10,0), padx=10)
    row += 1
    
    fields = [
        ("Dynamic PET Data:", var_data, False),
        ("ROI Voxel Idx:", var_roi, False),
        ("Estimates Output:", var_est_out, True)
    ]
    for label, var, is_save in fields:
        tk.Label(root, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(root, textvariable=var, width=50).grid(row=row, column=1, padx=5, pady=2)
        tk.Button(root, text="Browse", command=lambda v=var, s=is_save: browse_file(v, s)).grid(row=row, column=2, padx=5, pady=2)
        row += 1

    tk.Checkbutton(root, text="Save full posterior samples", variable=var_save_post).grid(row=row, column=1, sticky="w")
    row += 1

    # Imaging Section
    tk.Label(root, text="Imaging Settings", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", pady=(10,0), padx=10)
    row += 1
    
    fields_img = [
        ("Estimates Path:", var_est_in, False),
        ("Mask Path:", var_mask, False),
        ("Imaging Output:", var_img_out, True)
    ]
    for label, var, is_save in fields_img:
        tk.Label(root, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(root, textvariable=var, width=50).grid(row=row, column=1, padx=5, pady=2)
        tk.Button(root, text="Browse", command=lambda v=var, s=is_save: browse_file(v, s)).grid(row=row, column=2, padx=5, pady=2)
        row += 1

    # Dropdown & Slice Entry
    tk.Label(root, text="Parameter:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
    tk.OptionMenu(root, var_param, 'Ki', 'K1', 'k2', 'k3', 'k4').grid(row=row, column=1, sticky="w", padx=5, pady=2)
    row += 1
    
    tk.Label(root, text="Slice Index:").grid(row=row, column=0, sticky="e", padx=5, pady=2)
    tk.Entry(root, textvariable=var_slice, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=2)
    row += 1

    # Run Button Action
    def on_run():
        # Build the pipeline list based on checkboxes
        selected_pipeline = []
        if var_inf.get(): selected_pipeline.append('Inference')
        if var_img.get(): selected_pipeline.append('Imaging')
        
        if not selected_pipeline:
            messagebox.showwarning("Warning", "Please select at least one task to run.")
            return

        # Destroy GUI to unblock the main thread, then run the pipeline
        root.destroy()
        execute_pipeline(
            selected_pipeline,
            var_data.get(), var_roi.get(), var_est_out.get(), var_save_post.get(),
            var_est_in.get(), var_mask.get(), var_img_out.get(), var_param.get(), var_slice.get()
        )

    tk.Button(root, text="Run Pipeline", command=on_run, bg="green", fg="white", font=("Arial", 10, "bold")).grid(row=row, column=1, pady=20)

    # Start the GUI loop
    root.mainloop()

### ------ 4. Application Entry Point ------

if __name__ == '__main__':
    if USE_GUI:
        try:
            # Will throw TclError if running via SSH/HPC without a configured display
            launch_gui()
        except tk.TclError:
            print("No display detected. Falling back to terminal mode...")
            execute_pipeline(Run_pipeline, data_path, ROI_voxel_idx_path, estimates_output_path, 
                             save_full_posterior, estimates_path, mask_path, 
                             imaging_output_path, parameter_name, slice_index)
    else:
        print("GUI disabled. Falling back to terminal mode...")
        execute_pipeline(Run_pipeline, data_path, ROI_voxel_idx_path, estimates_output_path, 
                         save_full_posterior, estimates_path, mask_path, 
                         imaging_output_path, parameter_name, slice_index)