import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation

def plot_basic_results(results, params, save_path=None):
    plt.figure(figsize=(15, 12))

    # 1. Biomass
    plt.subplot(2, 2, 1)
    plt.plot(results['t'], results['I'], 'k-', label='Inert biomass (I)')
    plt.plot(results['t'], results['A'], 'b-', label='Down-regulated biomass (A)')
    plt.plot(results['t'], results['B'], 'g-', label='Up-regulated biomass (B)')
    plt.plot(results['t'], results['total_active'], 'r--', label='Total active biomass (A+B)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Biomass volume fraction')
    plt.title('Biomass change over time')
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

    # 3. Down-regulated ratio and active ratio
    plt.subplot(2, 2, 3)
    plt.plot(results['t'], results['Z'], 'b-', label='Down-regulated biomass ratio (Z)')
    plt.plot(results['t'], results['R'], 'g-', label='Active biomass ratio (R)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Ratio')
    plt.ylim(0, 1.1)
    plt.title('Down-regulated biomass ratio and active biomass ratio')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. Nutrients, antibiotics and biofilm net growth rate
    plt.subplot(2, 2, 4)
    plt.plot(results['t'], results['N'], 'g-', label='Nutrient concentration')
    plt.plot(results['t'], results['C'], 'r-', label='Antibiotic concentration')

    if 'BG' in results:
        plt.plot(results['t'], results['BG'], 'b--', label='Biofilm net growth rate')

    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Concentration/Growth rate')
    plt.title('Nutrient, antibiotic concentration and biofilm net growth rate')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_spatial_results(results, params, time_point, save_path=None):
    time_idx = np.argmin(np.abs(results['t'] - time_point))
    actual_time = results['t'][time_idx]

    plt.figure(figsize=(12, 10))

    n_layers = results['I_layers'].shape[1]
    depths = np.linspace(0, 1, n_layers)

    # 1. Biomass distribution
    plt.subplot(2, 2, 1)
    plt.plot(depths, results['I_layers'][time_idx], 'k-', label='Inert biomass (I)')
    plt.plot(depths, results['A_layers'][time_idx], 'b-', label='Down-regulated biomass (A)')
    plt.plot(depths, results['B_layers'][time_idx], 'g-', label='Up-regulated biomass (B)')
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

def plot_parameter_sensitivity(results, parameter_name, save_path=None):
    plt.figure(figsize=(15, 12))

    param_values = sorted(list(results.keys()))

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
            # Calculate if not provided
            threshold_idx = np.argmax(result['S'] > result['params'].tau)
            if threshold_idx > 0:
                time_to_upregulation.append(result['t'][threshold_idx])
            else:
                time_to_upregulation.append(np.nan)  # No upregulation occurred

        # Calculate resistance level (e.g., proportion of biomass surviving after antibiotic addition)
        if 'resistance_level' in result:
            resistance_level.append(result['resistance_level'])
        else:
            # Calculate if not provided
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
    plt.axhline(y=results[param_values[0]]['params'].tau, color='gray', linestyle='--', label='Base QS threshold')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration')
    plt.title('AHL concentration for different parameter values')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

def plot_results_with_qsi(results, params, save_path=None):
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 2, 5)
    plt.plot(results['t'], results['Q'], 'purple', label='QSI concentration')
    plt.axvline(x=params.T_QSI if params.T_QSI is not None else float('inf'),
                color='gray', linestyle='--', label='QSI addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('QSI concentration')
    plt.title('QSI concentration over time')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    plt.subplot(3, 2, 6)

    # Calculate QSI inhibition coefficient over time
    inhibition = results['Q'] / (params.K_Q + results['Q'])
    plt.plot(results['t'], 1 - inhibition, 'c-', label='AHL production efficiency')

    plt.xlabel('Dimensionless time')
    plt.ylabel('Efficiency factor')
    plt.title('AHL production efficiency under QSI influence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()