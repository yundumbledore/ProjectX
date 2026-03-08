import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def parametric_map_interactive(para_array, mask_slice, spacing, parameter_name, save_path, cbar=True):
    para_map = np.zeros_like(mask_slice, dtype=float)
    para_map[mask_slice == 1] = para_array

    index_map = np.zeros_like(mask_slice, dtype=int)
    index_map[mask_slice == 1] = np.arange(len(para_array))
    
    # --- START: Median Filtering ---
    
    # Define the neighborhood size. 
    # size=3 means a 3x3 kernel (looks at 8 neighbors).
    # size=5 means a 5x5 kernel (stronger effect). Start with 3.
    kernel_size = 2

    # Apply the median filter
    para_map = median_filter(para_map, size=kernel_size)
    
    # Re-apply the original mask to ensure no values "bleed" into the background
    para_map[mask_slice == 0] = 0.0
    
    # --- END: Median Filtering ---
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    # choose cmap and vmax/vmin as before…
    cmap = 'gray'
    if parameter_name == "Ki":
        vmax = 0.030; cmap = plt.cm.inferno
    elif parameter_name == "Ki Uncertainty":
        vmax = 0.12; cmap = plt.cm.inferno
    elif parameter_name == "Ki Posterior Std":
        vmax = 0.03; cmap = plt.cm.inferno
    elif parameter_name == "K_1":
        vmax = 0.9; cmap = plt.cm.inferno
    elif parameter_name == "Vb":
        vmax = np.max(para_map)
    elif parameter_name == "k_4":
        vmax = 0.125; cmap = plt.cm.inferno
    elif parameter_name == "k_2":
        vmax = 1.8; cmap = plt.cm.inferno
    elif parameter_name == "k_3":
        vmax = 0.3; cmap = plt.cm.inferno
    elif parameter_name == "EDT":
        vmax = 40
    elif parameter_name == "Patlak Ki":
        vmax = 0.02; cmap = plt.cm.inferno
    elif parameter_name == "Logan VT":
        vmax = 4
    elif parameter_name == "Coefficient of Variation":
        vmax = np.max(para_map); cmap = plt.cm.inferno
    elif parameter_name == "k4 Irreversibility":
        vmax = np.max(para_map)
        cmap = plt.cm.binary
    else:
        vmax = np.max(para_map)
    vmin = np.min(para_map)
    
    img_overlay = ax.imshow(
        para_map,
        cmap=cmap,
        alpha=1,
        aspect=spacing[1]/spacing[0], 
        vmax=vmax,
        vmin=vmin,
        interpolation='nearest'
    )
    img_overlay.set_clim(0, vmax)
    plt.axis('off')
    ax.invert_yaxis()

    if cbar:
        cbar = plt.colorbar(img_overlay, ax=ax, orientation='vertical')
        cbar.set_label('{}'.format(parameter_name), fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    # 4) override the status‐bar text
    def format_coord(x, y):
        """Given float x,y in data coords, convert to row, col, then lookup."""
        col = int(np.round(x))
        row = int(np.round(y))
        nrows, ncols = para_map.shape
        if (0 <= row < nrows) and (0 <= col < ncols):
            mask_val = mask_slice[row, col]
            if mask_val:
                idx = index_map[row, col]
                val = para_map[row, col]
                return (f'x={col}, y={row}   '
                        f'para_array index={idx}   '
                        f'value={val:.4g}')
        # fallback
        return f'x={col}, y={row}'

    ax.format_coord = format_coord

    plt.savefig(save_path, dpi=300)
    plt.show()
    
def computeKi(K1, k2, k3):
    return (K1 * k3)/(k2 + k3)

def parametric_imaging(slice_index, mask_path, estimates_path, parameter_name, save_path):
    mask = np.load(mask_path)['my_array']
    mask_slice = mask[:,slice_index,:]

    # This is P001 spacing. Try using it for all other subjects. Need to update if the image ratio looks strange.
    spacing = np.array([3.3,3.3,1.645])[[0,2]] 

    estimates = np.load(estimates_path)['arr']

    if parameter_name == 'Ki':
        statistics = estimates[:,0]
    elif parameter_name == 'K1':
        K1 = estimates[:,:,0]
        statistics = K1.mean(axis = 1)
    elif parameter_name == 'k2':
        k2 = estimates[:,:,1]
        statistics = k2.mean(axis = 1)
    elif parameter_name == 'k3':
        k3 = estimates[:,:,2]
        statistics = k3.mean(axis = 1)
    elif parameter_name == 'k4':
        k4 = estimates[:,:,3]
        statistics = k4.mean(axis = 1)

    parametric_map_interactive(statistics, mask_slice, spacing, parameter_name, save_path)