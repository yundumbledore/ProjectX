import sys
import os
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import warnings; warnings.filterwarnings("ignore")

# Ensure local sources are discoverable
dir_path = os.path.join(os.getcwd(), 'Sources')
sys.path.append(dir_path)

# Mock imports - ensure main.py exists in ./Sources
try:
    from main import run_inference, run_imaging
except ImportError:
    # Fallback for demonstration if main.py isn't found
    def run_inference(*args): print(f"DEBUG: Running Inference with {args}")
    def run_imaging(*args): print(f"DEBUG: Running Imaging with {args}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

### ------ 1. Default Settings ------
USE_GUI = False 
Run_pipeline = ['Inference', 'Imaging']

data_path = './Example_Data/P001_PET_incl_bones_coronal_116_denoised_adjusted.h5'
ROI_voxel_idx_path = './Example_Data/P001_coronal_116_ROI_voxel_idx.npz'
estimates_output_path = './Results/P001_PET_incl_bones_coronal_116_Ki.npz'
save_full_posterior = False 

estimates_path = './Results/P001_PET_incl_bones_coronal_116_Ki.npz'
mask_path = './Example_Data/P001_mask.npz'
imaging_output_path = './Results/P001_coronal_116_Ki.png'
parameter_name = 'Ki'
slice_index = 116

### ------ 2. Execution Logic ------

def execute_pipeline(pipeline, inf_data, inf_roi, inf_out, inf_save, 
                     img_est, img_mask, img_out, img_param, img_slice):
    """Runs the actual PyTorch tasks."""
    print(f"\n--- Starting execution on {device} ---")
    
    if 'Inference' in pipeline:
        print('Status: Running Inference...')
        run_inference(inf_data, inf_roi, inf_out, inf_save, device)
        
    if 'Imaging' in pipeline:
        print('Status: Running Imaging...')
        # Ensure slice is an integer
        run_imaging(int(img_slice), img_mask, img_est, img_param, img_out)
        
    print("--- All tasks completed successfully! ---\n")

### ------ 3. GUI Definition ------

def launch_gui():
    root = tk.Tk()
    root.title("Dynamic PET Analysis Configuration")
    root.geometry("650x550")

    # Variables
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

    def browse_file(var, is_save=False):
        if is_save:
            filepath = filedialog.asksaveasfilename()
        else:
            filepath = filedialog.askopenfilename()
        if filepath:
            var.set(filepath)

    # --- Layout ---
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    # Task Selection
    tk.Label(main_frame, text="Step 1: Select Tasks", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w")
    tk.Checkbutton(main_frame, text="Run Inference", variable=var_inf).grid(row=1, column=0, sticky="w", padx=10)
    tk.Checkbutton(main_frame, text="Run Imaging", variable=var_img).grid(row=1, column=1, sticky="w")

    ttk.Separator(main_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)

    # Inference Section
    tk.Label(main_frame, text="Step 2: Inference Configuration", font=("Arial", 11, "bold")).grid(row=3, column=0, sticky="w")
    
    inf_fields = [
        ("Input H5 Data:", var_data),
        ("ROI Voxel Idx:", var_roi),
        ("Output NPZ Path:", var_est_out)
    ]
    
    current_row = 4
    for label, var in inf_fields:
        tk.Label(main_frame, text=label).grid(row=current_row, column=0, sticky="e", pady=2)
        tk.Entry(main_frame, textvariable=var, width=45).grid(row=current_row, column=1, padx=5)
        tk.Button(main_frame, text="Browse", command=lambda v=var: browse_file(v)).grid(row=current_row, column=2)
        current_row += 1
    
    tk.Checkbutton(main_frame, text="Save full posterior samples", variable=var_save_post).grid(row=current_row, column=1, sticky="w")
    current_row += 1

    ttk.Separator(main_frame, orient='horizontal').grid(row=current_row, column=0, columnspan=3, sticky="ew", pady=10)
    current_row += 1

    # Imaging Section
    tk.Label(main_frame, text="Step 3: Imaging Configuration", font=("Arial", 11, "bold")).grid(row=current_row, column=0, sticky="w")
    current_row += 1

    img_fields = [
        ("Estimates NPZ:", var_est_in),
        ("Mask NPY Path:", var_mask),
        ("Output PNG Path:", var_img_out)
    ]

    for label, var in img_fields:
        tk.Label(main_frame, text=label).grid(row=current_row, column=0, sticky="e", pady=2)
        tk.Entry(main_frame, textvariable=var, width=45).grid(row=current_row, column=1, padx=5)
        tk.Button(main_frame, text="Browse", command=lambda v=var: browse_file(v)).grid(row=current_row, column=2)
        current_row += 1

    # Params
    tk.Label(main_frame, text="Parameter:").grid(row=current_row, column=0, sticky="e")
    tk.OptionMenu(main_frame, var_param, 'Ki', 'K1', 'k2', 'k3', 'k4').grid(row=current_row, column=1, sticky="w", padx=5)
    
    tk.Label(main_frame, text="Slice:").grid(row=current_row, column=1, sticky="e")
    tk.Entry(main_frame, textvariable=var_slice, width=5).grid(row=current_row, column=1, sticky="e", padx=(0, 40))
    current_row += 1

    def on_run():
        pipeline = []
        if var_inf.get(): pipeline.append('Inference')
        if var_img.get(): pipeline.append('Imaging')
        
        if not pipeline:
            messagebox.showwarning("Warning", "Please select at least one task.")
            return

        # Capture data before closing window
        args = (
            pipeline,
            var_data.get(), var_roi.get(), var_est_out.get(), var_save_post.get(),
            var_est_in.get(), var_mask.get(), var_img_out.get(), var_param.get(), var_slice.get()
        )
        
        # Hide window to show the user we are moving to processing
        root.withdraw()
        
        try:
            execute_pipeline(*args)
            messagebox.showinfo("Done", "Processing complete. Check terminal for logs.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            root.quit()
            root.destroy()

    tk.Button(main_frame, text="START PIPELINE", command=on_run, bg="#28a745", fg="white", 
              font=("Arial", 12, "bold"), pady=10, width=20).grid(row=current_row+1, column=0, columnspan=3, pady=20)

    root.mainloop()

### ------ 4. Application Entry Point ------

if __name__ == '__main__':
    if USE_GUI:
        try:
            launch_gui()
        except tk.TclError:
            print("No display detected. Falling back to terminal mode...")
            execute_pipeline(Run_pipeline, data_path, ROI_voxel_idx_path, estimates_output_path, 
                             save_full_posterior, estimates_path, mask_path, 
                             imaging_output_path, parameter_name, slice_index)
    else:
        execute_pipeline(Run_pipeline, data_path, ROI_voxel_idx_path, estimates_output_path, 
                         save_full_posterior, estimates_path, mask_path, 
                         imaging_output_path, parameter_name, slice_index)