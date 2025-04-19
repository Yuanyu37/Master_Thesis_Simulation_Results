import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Ensure that other modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.parameters import Parameters
from models.core_model import run_simulation
from utils.visualization import plot_basic_results

def run_experiment1_final():
    print("Running basic explanatory simulation...")
    # Initialize parameters
    params = Parameters()

    # Initial conditions
    I0 = 0.0  # Initial inert biomass
    A0 = 0.05  # Initial downregulated biomass
    B0 = 0.0  # Initial upregulated biomass
    N0 = 1.0  # Initial nutrient concentration
    C0 = 0.0  # Initial antibiotic concentration
    S0 = 0.0  # Initial AHL concentration

    initial_conditions = [I0, A0, B0, N0, C0, S0]

    # Run simulation
    results = run_simulation(params, initial_conditions, t_span=(0, 60), t_points=500)

    if results is None:
        print("Simulation failed!")
        return None

    # Plot results
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

    # 3. Downregulation and activity ratios
    plt.subplot(2, 2, 3)
    plt.plot(results['t'], results['Z'], 'b-', label='Downregulated biomass ratio (Z)')
    plt.plot(results['t'], results['R'], 'g-', label='Active biomass ratio (R)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Ratio')
    plt.ylim(0, 1.1)  # Ensure y-axis range is reasonable
    plt.title('Downregulated biomass ratio and active biomass ratio')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. Nutrients, antibiotics, and biofilm net growth rate
    plt.subplot(2, 2, 4)
    plt.plot(results['t'], results['N'], 'g-', label='Nutrient concentration')
    plt.plot(results['t'], results['C'], 'r-', label='Antibiotic concentration')
    plt.plot(results['t'], results['BG'], 'b--', label='Biofilm net growth rate')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Concentration/Growth rate')
    plt.title('Nutrient, antibiotic concentration, and biofilm net growth rate')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig('simulation_results/experiment1_final_results.png', dpi=300)
    plt.show()

    print("Basic explanatory simulation complete, results saved.")

    return results

def run_timing_experiment():
    print("Running antibiotic addition timing experiment...")

    # Different biofilm sizes for antibiotic addition
    biofilm_volumes = [0.05, 0.10, 0.15]  # Biofilm volume fractions (5%, 10%, 15%)

    # Simulation parameters
    params = Parameters()
    params.tau = 10.0  # Set lower QS activation threshold to accelerate upregulation
    params.C_inf = 2.0  # Antibiotic concentration

    # Reduce initial biomass
    initial_biomass = 0.01  # Set initial active biomass lower

    # First run a reference simulation without antibiotic addition
    print("Running reference simulation to determine biofilm growth characteristics...")
    reference_params = Parameters()
    reference_params.T_antibiotic = float('inf')  # No antibiotic addition
    reference_params.tau = 10.0  # Use the same QS threshold

    # Use lower initial biomass
    initial_conditions = [0.0, initial_biomass, 0.0, 1.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0]

    # Extend reference simulation time
    reference_results = run_simulation(reference_params, initial_conditions, t_span=(0, 100), t_points=1000)

    if reference_results is None:
        print("Reference simulation failed! Unable to determine antibiotic addition time.")
        return None

    # Print maximum biofilm size from reference simulation
    max_biomass = np.max(reference_results['total_biomass'])
    print(f"Maximum biofilm size in reference simulation: {max_biomass:.4f}")

    # Check and print initial biomass
    initial_biomass_simulated = reference_results['total_biomass'][0]
    print(f"Initial biofilm size: {initial_biomass_simulated:.4f}")

    antibiotic_times = {}

    # Scenario 1: Set antibiotic addition time based on biofilm growth curve
    # Find the time when biofilm reaches maximum size
    max_biomass_time = reference_results['t'][np.argmax(reference_results['total_biomass'])]
    print(f"Time when biofilm reaches maximum size: {max_biomass_time:.2f}")

    # Set early, middle, and late time points
    antibiotic_times[0.05] = max_biomass_time * 0.2  # Early growth
    antibiotic_times[0.10] = max_biomass_time * 0.5  # Mid growth
    antibiotic_times[0.15] = max_biomass_time * 0.8  # Late growth

    # Print set time points
    for volume, time in antibiotic_times.items():
        # Find the actual biofilm size at this time
        time_idx = np.argmin(np.abs(reference_results['t'] - time))
        actual_size = reference_results['total_biomass'][time_idx]
        print(f"Set time point: t = {time:.2f}, corresponding biofilm size: {actual_size:.4f}")

    # Run simulations at different time points
    results = {}

    # Run simulations for each time point
    for volume, ab_time in antibiotic_times.items():
        label = f"{volume * 100:.0f}% time point"
        print(f"Running simulation - {label}, antibiotic addition time: {ab_time:.2f}")

        # Set parameters
        sim_params = Parameters()
        sim_params.tau = 10.0
        sim_params.T_antibiotic = ab_time

        # Use the same initial conditions
        sim_results = run_simulation(sim_params, initial_conditions, t_span=(0, max_biomass_time + 40), t_points=500)

        if sim_results is None:
            print(f"Simulation failed at time point {ab_time:.2f}!")
            continue

        # Save results
        results[volume] = sim_results

        # Plot individual result
        save_path = f'simulation_results/timing_experiment_vol{volume * 100:.0f}.png'
        plot_basic_results(sim_results, sim_params, save_path)

    # Compare results at different time points
    if len(results) > 0:
        plt.figure(figsize=(15, 10))

        # 1. Compare total active biomass
        plt.subplot(2, 2, 1)
        for volume, result in sorted(results.items()):
            label = f'Volume={volume * 100:.0f}% time point'
            plt.plot(result['t'], result['total_active'], label=label)
            plt.axvline(x=antibiotic_times[volume], color='gray', linestyle='--')

        plt.xlabel('Dimensionless time')
        plt.ylabel('Total active biomass')
        plt.title('Total active biomass at different time points')
        plt.legend()
        plt.grid(True)

        # 2. Compare AHL concentration
        plt.subplot(2, 2, 2)
        for volume, result in sorted(results.items()):
            label = f'Volume={volume * 100:.0f}% time point'
            plt.plot(result['t'], result['S'], label=label)
            plt.axvline(x=antibiotic_times[volume], color='gray', linestyle='--')

        plt.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
        plt.xlabel('Dimensionless time')
        plt.ylabel('AHL concentration [nM]')
        plt.title('AHL concentration at different time points')
        plt.legend()
        plt.grid(True)

        # 3. Compare active ratio
        plt.subplot(2, 2, 3)
        for volume, result in sorted(results.items()):
            label = f'Volume={volume * 100:.0f}% time point'
            plt.plot(result['t'], result['R'], label=label)
            plt.axvline(x=antibiotic_times[volume], color='gray', linestyle='--')

        plt.xlabel('Dimensionless time')
        plt.ylabel('Active ratio')
        plt.title('Active ratio at different time points')
        plt.legend()
        plt.grid(True)

        # 4. Compare downregulated ratio
        plt.subplot(2, 2, 4)
        for volume, result in sorted(results.items()):
            label = f'Volume={volume * 100:.0f}% time point'
            plt.plot(result['t'], result['Z'], label=label)
            plt.axvline(x=antibiotic_times[volume], color='gray', linestyle='--')

        plt.xlabel('Dimensionless time')
        plt.ylabel('Downregulated ratio')
        plt.title('Downregulated ratio at different time points')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('simulation_results/timing_experiment_comparison.png', dpi=300)
        plt.show()

    print("Antibiotic addition timing experiment complete, results saved.")

    return results

def run_always_up_down_experiment():
    print("Running always up, adaptive, and always down comparison experiment...")

    results = {}

    # Case 1: Always downregulated (no QS)
    print("Simulating Case 1: Always downregulated...")
    params1 = Parameters()
    params1.omega = 0.0  # Disable upregulation
    params1.psi = 0.0  # Disable downregulation

    # Set initial conditions to all downregulated biomass
    initial_conditions1 = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0]

    results1 = run_simulation(params1, initial_conditions1)
    if results1 is not None:
        results['case1'] = results1

    # Case 2: Adaptive QS regulation
    print("Simulating Case 2: Adaptive QS regulation...")
    params2 = Parameters()
    params2.tau = 10.0  # Set lower QS activation threshold

    # Set initial conditions to all downregulated biomass
    initial_conditions2 = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0]

    results2 = run_simulation(params2, initial_conditions2)
    if results2 is not None:
        results['case2'] = results2

    # Case 3: Always upregulated
    print("Simulating Case 3: Always upregulated...")
    params3 = Parameters()
    params3.omega = 0.0  # Disable upregulation
    params3.psi = 0.0  # Disable downregulation

    # Set initial conditions to all upregulated biomass
    initial_conditions3 = [0.0, 0.0, 0.05, 1.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0]

    results3 = run_simulation(params3, initial_conditions3)
    if results3 is not None:
        results['case3'] = results3

    # Plot comparison results
    plt.figure(figsize=(15, 12))

    # 1. Compare total active biomass
    plt.subplot(2, 2, 1)
    labels = {
        'case1': 'Always downregulated',
        'case2': 'Adaptive QS regulation',
        'case3': 'Always upregulated'
    }

    for case, result in results.items():
        plt.plot(result['t'], result['total_active'], label=labels[case])
    plt.axvline(x=params1.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Total active biomass')
    plt.title('Total active biomass in different cases')
    plt.legend()
    plt.grid(True)

    # 2. Compare individual biomass fractions
    plt.subplot(2, 2, 2)
    if 'case1' in results:
        plt.plot(results['case1']['t'], results['case1']['A'], 'b-', label='Downregulated (A) - Always downregulated')
    if 'case2' in results:
        plt.plot(results['case2']['t'], results['case2']['A'], 'b--', label='Downregulated (A) - Adaptive')
        plt.plot(results['case2']['t'], results['case2']['B'], 'g--', label='Upregulated (B) - Adaptive')
    if 'case3' in results:
        plt.plot(results['case3']['t'], results['case3']['B'], 'g-', label='Upregulated (B) - Always upregulated')
    plt.axvline(x=params1.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Biomass volume fraction')
    plt.title('Biomass fractions in different cases')
    plt.legend()
    plt.grid(True)

    # 3. Compare AHL concentration
    plt.subplot(2, 2, 3)
    for case, result in results.items():
        plt.plot(result['t'], result['S'], label=f'AHL - {labels[case]}')
    plt.axhline(y=params2.tau, color='gray', linestyle='--', label='QS threshold')
    plt.axvline(x=params1.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration [nM]')
    plt.title('AHL concentration in different cases')
    plt.legend()
    plt.grid(True)

    # 4. Compare active ratio
    plt.subplot(2, 2, 4)
    for case, result in results.items():
        plt.plot(result['t'], result['R'], label=f'Active ratio - {labels[case]}')
    plt.axvline(x=params1.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Active ratio')
    plt.title('Active ratio in different cases')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_results/always_up_down_comparison.png', dpi=300)
    plt.show()

    print("Always up, adaptive, and always down comparison experiment complete, results saved.")

    return results
