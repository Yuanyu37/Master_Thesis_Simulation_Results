import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation

def plot_basic_results(results, params, save_path=None):
    plt.figure(figsize=(15, 12))

    # 1. Biomass
    plt.subplot(2, 2, 1)
    plt.plot(results['t'], results['I'], 'k-', label='Inert biomass (I)')
    plt.plot(results['t'], results['A'], 'b-', label='Downregulated biomass (A)')
    plt.plot(results['t'], results['B'], 'g-', label='Upregulated biomass (B)')
    plt.plot(results['t'], results['total_active'], 'r--', label='Total active biomass (A+B)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Biomass volume fraction')
    plt.title('Biomass over time')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. AHL concentration
    plt.subplot(2, 2, 2)
    plt.plot(results['t'], results['S'], 'r-', label='AHL concentration')
    plt.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration [nM]')
    plt.title('AHL concentration over time')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. Downregulated and active ratios
    plt.subplot(2, 2, 3)
    plt.plot(results['t'], results['Z'], 'b-', label='Downregulated biomass ratio (Z)')
    plt.plot(results['t'], results['R'], 'g-', label='Active biomass ratio (R)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Ratio')
    plt.ylim(0, 1.1)  # Ensure the y-axis range is reasonable
    plt.title('Downregulated biomass ratio and active biomass ratio')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. Nutrients, antibiotics, and biofilm net growth rate
    plt.subplot(2, 2, 4)
    plt.plot(results['t'], results['N'], 'g-', label='Nutrient concentration')
    plt.plot(results['t'], results['C'], 'r-', label='Antibiotic concentration')

    if 'BG' in results:
        plt.plot(results['t'], results['BG'], 'b--', label='Biofilm net growth rate')

    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Concentration/Growth rate')
    plt.title('Nutrient, antibiotic concentration, and biofilm net growth rate')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_spatial_results(results, params, time_point, save_path=None):
    # Find the index closest to the specified time point
    time_idx = np.argmin(np.abs(results['t'] - time_point))
    actual_time = results['t'][time_idx]

    plt.figure(figsize=(12, 10))

    # Get the number of layers
    n_layers = results['I_layers'].shape[1]
    depths = np.linspace(0, 1, n_layers)

    # 1. Biomass distribution
    plt.subplot(2, 2, 1)
    plt.plot(depths, results['I_layers'][time_idx], 'k-', label='Inert biomass (I)')
    plt.plot(depths, results['A_layers'][time_idx], 'b-', label='Downregulated biomass (A)')
    plt.plot(depths, results['B_layers'][time_idx], 'g-', label='Upregulated biomass (B)')
    plt.xlabel('Relative depth')
    plt.ylabel('Biomass volume fraction')
    plt.title(f'Biomass distribution (t={actual_time:.2f})')
    plt.legend()
    plt.grid(True)

    # 2. AHL distribution
    plt.subplot(2, 2, 2)
    plt.plot(depths, results['S_layers'][time_idx], 'r-')
    plt.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
    plt.xlabel('Relative depth')
    plt.ylabel('AHL concentration [nM]')
    plt.title(f'AHL distribution (t={actual_time:.2f})')
    plt.legend()
    plt.grid(True)

    # 3. Nutrient distribution
    plt.subplot(2, 2, 3)
    plt.plot(depths, results['N_layers'][time_idx], 'g-')
    plt.xlabel('Relative depth')
    plt.ylabel('Nutrient concentration')
    plt.title(f'Nutrient distribution (t={actual_time:.2f})')
    plt.grid(True)

    # 4. Antibiotic distribution
    plt.subplot(2, 2, 4)
    plt.plot(depths, results['C_layers'][time_idx], 'r-')
    plt.xlabel('Relative depth')
    plt.ylabel('Antibiotic concentration')
    plt.title(f'Antibiotic distribution (t={actual_time:.2f})')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_parameter_variation(results, parameter_name, save_path=None):
    plt.figure(figsize=(15, 12))

    param_values = sorted(list(results.keys()))

    # Extract results
    final_biomass = []
    time_to_upregulation = []
    resistance_level = []

    for val in param_values:
        result = results[val]
        final_biomass.append(result['total_active'][-1])

        # Find upregulation time (time when AHL exceeds threshold)
        if 'time_to_upregulation' in result:
            time_to_upregulation.append(result['time_to_upregulation'])
        else:
            # If not provided, calculate it
            threshold_idx = np.argmax(result['S'] > result['params'].tau)
            if threshold_idx > 0:
                time_to_upregulation.append(result['t'][threshold_idx])
            else:
                time_to_upregulation.append(np.nan)  # No upregulation

        # Calculate resistance level (e.g., proportion of biomass surviving after antibiotic addition)
        if 'resistance_level' in result:
            resistance_level.append(result['resistance_level'])
        else:
            # If not provided, calculate it
            antibiotic_idx = np.argmax(result['t'] >= result['params'].T_antibiotic)
            if antibiotic_idx > 0:
                pre_antibiotic = result['total_active'][antibiotic_idx - 1]
                final = result['total_active'][-1]
                resistance_level.append(final / pre_antibiotic if pre_antibiotic > 0 else 0)
            else:
                resistance_level.append(np.nan)

    # 1. Final biomass
    plt.subplot(2, 2, 1)
    plt.plot(param_values, final_biomass, 'bo-')
    plt.xlabel(f'{parameter_name} value')
    plt.ylabel('Final active biomass')
    plt.title(f'Effect of {parameter_name} on final biomass')
    plt.grid(True)

    # 2. Upregulation time
    plt.subplot(2, 2, 2)
    plt.plot(param_values, time_to_upregulation, 'go-')
    plt.xlabel(f'{parameter_name} value')
    plt.ylabel('Upregulation time')
    plt.title(f'Effect of {parameter_name} on upregulation time')
    plt.grid(True)

    # 3. Resistance level
    plt.subplot(2, 2, 3)
    plt.plot(param_values, resistance_level, 'ro-')
    plt.xlabel(f'{parameter_name} value')
    plt.ylabel('Resistance level')
    plt.title(f'Effect of {parameter_name} on resistance level')
    plt.grid(True)

    # 4. AHL concentration time series comparison
    plt.subplot(2, 2, 4)
    for val in param_values:
        result = results[val]
        plt.plot(result['t'], result['S'], label=f'{parameter_name}={val}')
    plt.axhline(y=results[param_values[0]]['params'].tau, color='gray', linestyle='--', label='Baseline QS threshold')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration')
    plt.title('AHL concentration for different parameter values')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def create_biofilm_animation(results, params, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Determine the time step interval
    n_frames = min(100, len(results['t']))  # Maximum of 100 frames
    frame_indices = np.linspace(0, len(results['t']) - 1, n_frames, dtype=int)

    # Initialize plots
    line1_I, = ax1.plot([], [], 'k-', label='Inert biomass (I)')
    line1_A, = ax1.plot([], [], 'b-', label='Downregulated biomass (A)')
    line1_B, = ax1.plot([], [], 'g-', label='Upregulated biomass (B)')
    line1_total, = ax1.plot([], [], 'r--', label='Total active biomass (A+B)')
    time_line1 = ax1.axvline(x=0, color='purple', linestyle='-', alpha=0.5)
    ab_line1 = ax1.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')

    ax1.set_xlabel('Dimensionless time')
    ax1.set_ylabel('Biomass volume fraction')
    ax1.set_title('Biomass over time')
    ax1.set_xlim(results['t'][0], results['t'][-1])
    ax1.set_ylim(0, max(np.max(results['I']), np.max(results['A']), np.max(results['B']),
                        np.max(results['total_active'])) * 1.1)
    ax1.legend()
    ax1.grid(True)

    # AHL concentration
    line2, = ax2.plot([], [], 'r-', label='AHL concentration')
    time_line2 = ax2.axvline(x=0, color='purple', linestyle='-', alpha=0.5)
    threshold_line = ax2.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
    ab_line2 = ax2.axvline(x=params.T_antibiotic, color='gray', linestyle='--')

    ax2.set_xlabel('Dimensionless time')
    ax2.set_ylabel('AHL concentration [nM]')
    ax2.set_title('AHL concentration over time')
    ax2.set_xlim(results['t'][0], results['t'][-1])
    ax2.set_ylim(0, np.max(results['S']) * 1.1)
    ax2.legend()
    ax2.grid(True)

    line3_Z, = ax3.plot([], [], 'b-', label='Downregulated biomass ratio (Z)')
    line3_R, = ax3.plot([], [], 'g-', label='Active biomass ratio (R)')
    time_line3 = ax3.axvline(x=0, color='purple', linestyle='-', alpha=0.5)
    ab_line3 = ax3.axvline(x=params.T_antibiotic, color='gray', linestyle='--')

    ax3.set_xlabel('Dimensionless time')
    ax3.set_ylabel('Ratio')
    ax3.set_title('Downregulated biomass ratio and active biomass ratio')
    ax3.set_xlim(results['t'][0], results['t'][-1])
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(True)

    # Nutrients and antibiotics
    line4_N, = ax4.plot([], [], 'g-', label='Nutrient concentration')
    line4_C, = ax4.plot([], [], 'r-', label='Antibiotic concentration')
    if 'BG' in results:
        line4_BG, = ax4.plot([], [], 'b--', label='Biofilm net growth rate')
    time_line4 = ax4.axvline(x=0, color='purple', linestyle='-', alpha=0.5)
    ab_line4 = ax4.axvline(x=params.T_antibiotic, color='gray', linestyle='--')

    ax4.set_xlabel('Dimensionless time')
    ax4.set_ylabel('Concentration/Growth rate')
    ax4.set_title('Nutrient and antibiotic concentration')
    ax4.set_xlim(results['t'][0], results['t'][-1])
    ax4.set_ylim(0, max(np.max(results['N']), np.max(results['C'])) * 1.1)
    ax4.legend()
    ax4.grid(True)

    # Set title
    fig.suptitle('Biofilm quorum sensing and antibiotic response simulation', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Initialize function
    def init():
        line1_I.set_data([], [])
        line1_A.set_data([], [])
        line1_B.set_data([], [])
        line1_total.set_data([], [])
        line2.set_data([], [])
        line3_Z.set_data([], [])
        line3_R.set_data([], [])
        line4_N.set_data([], [])
        line4_C.set_data([], [])
        if 'BG' in results:
            line4_BG.set_data([], [])

        time_line1.set_xdata([0])
        time_line2.set_xdata([0])
        time_line3.set_xdata([0])
        time_line4.set_xdata([0])

        if 'BG' in results:
            return line1_I, line1_A, line1_B, line1_total, line2, line3_Z, line3_R, line4_N, line4_C, line4_BG, time_line1, time_line2, time_line3, time_line4
        else:
            return line1_I, line1_A, line1_B, line1_total, line2, line3_Z, line3_R, line4_N, line4_C, time_line1, time_line2, time_line3, time_line4

    # Update function
    def update(frame):
        idx = frame_indices[frame]
        current_time = results['t'][idx]

        # Update data
        line1_I.set_data(results['t'][:idx + 1], results['I'][:idx + 1])
        line1_A.set_data(results['t'][:idx + 1], results['A'][:idx + 1])
        line1_B.set_data(results['t'][:idx + 1], results['B'][:idx + 1])
        line1_total.set_data(results['t'][:idx + 1], results['total_active'][:idx + 1])

        line2.set_data(results['t'][:idx + 1], results['S'][:idx + 1])

        line3_Z.set_data(results['t'][:idx + 1], results['Z'][:idx + 1])
        line3_R.set_data(results['t'][:idx + 1], results['R'][:idx + 1])

        line4_N.set_data(results['t'][:idx + 1], results['N'][:idx + 1])
        line4_C.set_data(results['t'][:idx + 1], results['C'][:idx + 1])
        if 'BG' in results:
            line4_BG.set_data(results['t'][:idx + 1], results['BG'][:idx + 1])

        # Update current time line
        time_line1.set_xdata([current_time])
        time_line2.set_xdata([current_time])
        time_line3.set_xdata([current_time])
        time_line4.set_xdata([current_time])

        # Update title
        fig.suptitle(f'Biofilm quorum sensing and antibiotic response simulation (t = {current_time:.2f})', fontsize=16)

        if 'BG' in results:
            return line1_I, line1_A, line1_B, line1_total, line2, line3_Z, line3_R, line4_N, line4_C, line4_BG, time_line1, time_line2, time_line3, time_line4
        else:
            return line1_I, line1_A, line1_B, line1_total, line2, line3_Z, line3_R, line4_N, line4_C, time_line1, time_line2, time_line3, time_line4

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=100)

    # Save animation
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=10, dpi=200)

    plt.show()

    return ani
