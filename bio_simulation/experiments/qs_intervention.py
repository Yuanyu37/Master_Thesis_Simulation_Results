import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy

# Ensure other modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.parameters import Parameters
from models.core_model import run_simulation, ode_system
from utils.visualization import plot_basic_results, plot_parameter_variation

def run_qs_parameter_variation():
    print("Running quorum sensing parameter variation study...")

    # Parameters and values to study
    variation_params = {
        'tau': [5.0, 10.0, 20.0, 40.0],  # QS activation threshold
        'omega': [1.0, 2.0, 4.0, 8.0],  # Upregulation rate
        'sigma_S': [30.0, 100.0, 300.0, 600.0]  # AHL production rate in upregulated state
    }

    results = {}

    # Run variation analysis for each parameter
    for param_name, param_values in variation_params.items():
        print(f"Testing parameter: {param_name}")

        param_results = {}

        for value in param_values:
            print(f"  Testing value: {value}")

            # Initialize parameters
            params = Parameters()

            # Set parameter value
            if param_name == 'tau':
                params.tau = value
            elif param_name == 'omega':
                params.omega = value
            elif param_name == 'sigma_S':
                params.sigma_S = value

            # Run simulation
            sim_results = run_simulation(params)

            if sim_results is None:
                print(f"  Simulation failed for parameter value {value}!")
                continue

            # Add parameters to results
            sim_results['params'] = params

            # Calculate key metrics
            # Time to upregulation (time when AHL exceeds threshold)
            threshold_idx = np.argmax(sim_results['S'] > params.tau)
            if threshold_idx > 0:
                sim_results['time_to_upregulation'] = sim_results['t'][threshold_idx]
            else:
                sim_results['time_to_upregulation'] = float('inf')  # No upregulation occurred

            # Resistance level (proportion of biomass surviving after antibiotic addition)
            antibiotic_idx = np.argmax(sim_results['t'] >= params.T_antibiotic)
            if antibiotic_idx > 0:
                pre_antibiotic = sim_results['total_active'][antibiotic_idx - 1]
                final = sim_results['total_active'][-1]
                sim_results['resistance_level'] = final / pre_antibiotic if pre_antibiotic > 0 else 0
            else:
                sim_results['resistance_level'] = 0

            # Save individual result
            param_results[value] = sim_results

            # Plot individual result
            save_path = f'simulation_results/param_variation_{param_name}_{value}.png'
            plot_basic_results(sim_results, params, save_path)

        # Save all results for this parameter
        results[param_name] = param_results

        # Plot parameter variation comparison graphs
        plt.figure(figsize=(15, 10))

        # Extract results
        param_values_list = sorted(list(param_results.keys()))
        final_biomass = []
        time_to_upregulation = []
        resistance_level = []

        for val in param_values_list:
            result = param_results[val]
            final_biomass.append(result['total_active'][-1])
            time_to_upregulation.append(
                result['time_to_upregulation'] if result['time_to_upregulation'] != float('inf') else None)
            resistance_level.append(result['resistance_level'])

        # 1. Final biomass
        plt.subplot(2, 2, 1)
        plt.plot(param_values_list, final_biomass, 'bo-')
        plt.xlabel(f'{param_name} value')
        plt.ylabel('Final active biomass')
        plt.title(f'Effect of different {param_name} values on final biomass')
        plt.grid(True)

        # 2. Time to upregulation
        plt.subplot(2, 2, 2)
        plt.plot(param_values_list, time_to_upregulation, 'go-')
        plt.xlabel(f'{param_name} value')
        plt.ylabel('Time to upregulation')
        plt.title(f'Effect of different {param_name} values on upregulation time')
        plt.grid(True)

        # 3. Resistance level
        plt.subplot(2, 2, 3)
        plt.plot(param_values_list, resistance_level, 'ro-')
        plt.xlabel(f'{param_name} value')
        plt.ylabel('Resistance level')
        plt.title(f'Effect of different {param_name} values on resistance level')
        plt.grid(True)

        # 4. AHL concentration time series comparison
        plt.subplot(2, 2, 4)
        for val in param_values_list:
            result = param_results[val]
            plt.plot(result['t'], result['S'], label=f'{param_name}={val}')
        plt.axhline(y=params.tau, color='gray', linestyle='--', label='Baseline QS threshold')
        plt.xlabel('Dimensionless time')
        plt.ylabel('AHL concentration')
        plt.title(f'AHL concentration changes for different {param_name} values')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'simulation_results/param_variation_{param_name}_comparison.png', dpi=300)
        plt.show()

    print("Quorum sensing parameter variation study completed, results saved")

    return results

def evaluate_combined_strategies():
    """
    Evaluate combined strategies of antibiotics and QSI
    """
    print("Evaluating combined strategies...")

    # Define different strategy combinations
    strategies = [
        # [Antibiotic time, Antibiotic concentration, QSI time, QSI strength]
        [40.0, 2.0, None, 0.0],  # Antibiotics only
        [40.0, 0.0, 20.0, 5.0],  # QSI only
        [40.0, 2.0, 20.0, 5.0],  # QSI before antibiotics
        [40.0, 2.0, 40.0, 5.0],  # QSI simultaneous with antibiotics
        [20.0, 2.0, 40.0, 5.0],  # Antibiotics before QSI
    ]

    strategy_names = [
        "Antibiotics only",
        "QSI only",
        "QSI before antibiotics",
        "QSI simultaneous with antibiotics",
        "Antibiotics before QSI"
    ]

    # Define different line styles and colors for each strategy to ensure distinguishability
    line_styles = {
        "Antibiotics only": {"color": "blue", "linestyle": "-", "linewidth": 2.5},
        "QSI only": {"color": "orange", "linestyle": "--", "linewidth": 3, "marker": "o", "markevery": 30},
        "QSI before antibiotics": {"color": "green", "linestyle": "-", "linewidth": 2.5},
        "QSI simultaneous with antibiotics": {"color": "red", "linestyle": ":", "linewidth": 3, "marker": "^", "markevery": 30},
        "Antibiotics before QSI": {"color": "purple", "linestyle": "-", "linewidth": 2.5}
    }

    results = {}
    metrics = {}

    for i, (ab_time, ab_conc, qsi_time, qsi_strength) in enumerate(strategies):
        strategy_name = strategy_names[i]
        print(f"Evaluating strategy {i + 1}: {strategy_name}...")

        # Set parameters
        params = Parameters()
        params.T_antibiotic = ab_time
        params.C_inf = ab_conc

        # Run simulation
        if qsi_time is not None:
            sim_results = run_qsi_simulation(qsi_strength, qsi_time)
        else:
            sim_results = run_simulation(params)

        if sim_results is None:
            print(f"Simulation failed for strategy {strategy_name}!")
            continue

        # Save results
        results[strategy_name] = sim_results

        # Calculate performance metrics
        final_biomass = sim_results['total_active'][-1]
        initial_biomass = sim_results['total_active'][0]

        # Biomass before antibiotic application
        if ab_time is not None and ab_time > 0:
            ab_idx = np.argmax(sim_results['t'] >= ab_time)
            pre_antibiotic_biomass = sim_results['total_active'][ab_idx - 1] if ab_idx > 0 else initial_biomass
        else:
            pre_antibiotic_biomass = initial_biomass

        # Compute metrics
        metrics[strategy_name] = {
            'Final active biomass': final_biomass,
            'Biofilm eradication rate': 1 - (final_biomass / pre_antibiotic_biomass) if pre_antibiotic_biomass > 0 else 0,
            'Max AHL concentration': np.max(sim_results['S']),
            'Antibiotic usage': ab_conc * (sim_results['t'][-1] - ab_time) if ab_time is not None else 0,
            'Treatment duration': sim_results['t'][-1] - min(ab_time if ab_time is not None else float('inf'),
                                                            qsi_time if qsi_time is not None else float('inf'))
        }

    # Compare different strategies
    if len(results) > 0:
        fig = plt.figure(figsize=(16, 12))

        # 1. Total active biomass
        ax1 = plt.subplot(2, 2, 1)

        # Draw lines in a specific order to ensure visibility: draw main lines first, then those likely to be covered
        plot_order = ["Antibiotics before QSI", "QSI before antibiotics", "Antibiotics only", "QSI simultaneous with antibiotics", "QSI only"]

        for name in plot_order:
            if name in results:
                result = results[name]
                style = line_styles[name]
                ax1.plot(result['t'], result['total_active'], label=name, **style)

        # Mark treatment time points - use vertical lines to indicate key timings
        for name, result in results.items():
            idx = strategies[list(strategy_names).index(name)]
            if idx[0] is not None:  # Antibiotic addition time
                ax1.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.3)
            if idx[2] is not None:  # QSI addition time
                ax1.axvline(x=idx[2], color='purple', linestyle='--', alpha=0.3)

        ax1.set_xlabel('Dimensionless time')
        ax1.set_ylabel('Total active biomass')
        ax1.set_title('Total active biomass under different strategies')
        ax1.legend(loc='best')
        ax1.grid(True)

        # 2. AHL concentration
        ax2 = plt.subplot(2, 2, 2)

        for name in plot_order:
            if name in results:
                result = results[name]
                style = line_styles[name]
                ax2.plot(result['t'], result['S'], label=name, **style)

        # Draw QS threshold line
        base_params = Parameters()
        ax2.axhline(y=base_params.tau, color='gray', linestyle='--', label='QS threshold')

        # Mark treatment time points
        for name, result in results.items():
            idx = strategies[list(strategy_names).index(name)]
            if idx[0] is not None:
                ax2.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.3)
            if idx[2] is not None:
                ax2.axvline(x=idx[2], color='purple', linestyle='--', alpha=0.3)

        ax2.set_xlabel('Dimensionless time')
        ax2.set_ylabel('AHL concentration [nM]')
        ax2.set_title('AHL concentration under different strategies')
        ax2.legend(loc='best')
        ax2.grid(True)

        # 3. Antibiotic concentration
        ax3 = plt.subplot(2, 2, 3)

        for name in plot_order:
            if name in results:
                result = results[name]
                style = line_styles[name]

                # For the "QSI only" strategy, ensure zero antibiotic concentration is still visualized
                if name == "QSI only":
                    # Create an explicit zero concentration line
                    ax3.plot(result['t'], np.zeros_like(result['t']), label=name, **style)
                else:
                    ax3.plot(result['t'], result['C'], label=name, **style)

        # Mark treatment time points
        for name, result in results.items():
            idx = strategies[list(strategy_names).index(name)]
            if idx[0] is not None:
                ax3.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.3)
            if idx[2] is not None:
                ax3.axvline(x=idx[2], color='purple', linestyle='--', alpha=0.3)

        ax3.set_xlabel('Dimensionless time')
        ax3.set_ylabel('Antibiotic concentration')
        ax3.set_title('Antibiotic concentration under different strategies')
        ax3.legend(loc='best')
        ax3.grid(True)

        # 4. Performance metrics as bar charts with dual Y axes
        ax4 = plt.subplot(2, 2, 4)

        # Extract metrics for each strategy
        strategy_names_sorted = list(metrics.keys())
        x = np.arange(len(strategy_names_sorted))
        width = 0.35

        # Extract metric data
        eradication_rates = [metrics[name]['Biofilm eradication rate'] for name in strategy_names_sorted]
        final_biomass = [metrics[name]['Final active biomass'] for name in strategy_names_sorted]

        # Use dual Y axes to display different metrics
        bars1 = ax4.bar(x - width/2, eradication_rates, width, color='blue', label='Biofilm eradication rate')

        # Create second Y axis
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, final_biomass, width, color='orange', label='Final active biomass')

        # Set Y axis range to ensure small values are visible
        max_biomass = max(final_biomass) if max(final_biomass) > 0 else 0.01
        ax4_twin.set_ylim(0, max_biomass * 1.2)

        # Add grid lines for readability
        ax4.grid(True, alpha=0.3)

        # Add labels and legends
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Biofilm eradication rate', color='blue')
        ax4_twin.set_ylabel('Final active biomass', color='orange')
        ax4.set_title('Comparison of strategy effectiveness')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategy_names_sorted, rotation=45, ha='right')

        # Combine legends from both axes
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

        # Annotate bar values
        for bar in bars1:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax4_twin.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('simulation_results/combined_strategies_comparison.png', dpi=300)
        plt.show()

        # Create an additional log-scale plot to better show small value changes
        plt.figure(figsize=(10, 6))

        for name in plot_order:
            if name in results:
                result = results[name]
                style = line_styles[name]
                # Add a small constant to avoid log(0) error
                plt.semilogy(result['t'], result['total_active'] + 1e-10, label=name, **style)

        plt.xlabel('Dimensionless time')
        plt.ylabel('Total active biomass (log scale)')
        plt.title('Total active biomass under different strategies (log scale)')
        plt.grid(True, which="both", ls="-")
        plt.legend(loc='best')
        plt.savefig('simulation_results/combined_strategies_log_scale.png', dpi=300)
        plt.show()

    # Print detailed metrics
    print("\nComparison of strategy effectiveness metrics:")
    for name, m in metrics.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    print("\nCombined strategy evaluation complete. Results saved.")

    return {'results': results, 'metrics': metrics}

def analyze_stress_response():

    print("Analyzing stress response mechanism...")

    # Test effect of different antibiotic concentrations on AHL production
    ab_concentrations = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]

    results = {}
    metrics = {}

    for c in ab_concentrations:
        # Set parameters
        params = Parameters()
        params.C_inf = c
        params.T_antibiotic = 20.0  # Add antibiotic in early biofilm formation

        # Run simulation
        sim_results = run_simulation(params)

        if sim_results is None:
            print(f"Simulation failed for antibiotic concentration {c}!")
            continue

        # Save results
        results[c] = sim_results

        # Calculate key metrics
        max_ahl = np.max(sim_results['S'])

        # Find time to upregulation (time when AHL exceeds threshold)
        threshold_idx = np.argmax(sim_results['S'] > params.tau)
        if threshold_idx > 0:
            time_to_upregulation = sim_results['t'][threshold_idx]
        else:
            time_to_upregulation = float('inf')  # No upregulation occurred

        # Find stress response peak (AHL peak after antibiotic addition)
        ab_idx = np.argmax(sim_results['t'] >= params.T_antibiotic)
        if ab_idx > 0:
            stress_peak_idx = ab_idx + np.argmax(sim_results['S'][ab_idx:])
            stress_peak_ahl = sim_results['S'][stress_peak_idx]
            stress_peak_time = sim_results['t'][stress_peak_idx]
        else:
            stress_peak_ahl = 0
            stress_peak_time = 0

        # Record metrics
        metrics[c] = {
            'Maximum AHL concentration': max_ahl,
            'Time to upregulation': time_to_upregulation,
            'Stress peak AHL': stress_peak_ahl,
            'Stress peak time': stress_peak_time,
            'Final active biomass': sim_results['total_active'][-1]
        }

    # Compare different antibiotic concentrations
    plt.figure(figsize=(15, 10))

    # 1. Compare total active biomass
    plt.subplot(2, 2, 1)
    for c, result in results.items():
        plt.plot(result['t'], result['total_active'], label=f'C_inf={c}')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Total active biomass')
    plt.title('Total active biomass for different antibiotic concentrations')
    plt.legend()
    plt.grid(True)

    # 2. Compare AHL concentration
    plt.subplot(2, 2, 2)
    for c, result in results.items():
        plt.plot(result['t'], result['S'], label=f'C_inf={c}')
    plt.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration [nM]')
    plt.title('AHL concentration for different antibiotic concentrations')
    plt.legend()
    plt.grid(True)

    # 3. Relationship between antibiotic concentration and stress response
    plt.subplot(2, 2, 3)
    conc = list(metrics.keys())
    max_ahl = [metrics[c]['Maximum AHL concentration'] for c in conc]
    stress_peak = [metrics[c]['Stress peak AHL'] for c in conc]

    plt.plot(conc, max_ahl, 'bo-', label='Maximum AHL concentration')
    plt.plot(conc, stress_peak, 'ro-', label='Stress peak AHL')
    plt.xlabel('Antibiotic concentration')
    plt.ylabel('AHL concentration [nM]')
    plt.title('Relationship between antibiotic concentration and stress response')
    plt.legend()
    plt.grid(True)

    # 4. Relationship between antibiotic concentration and upregulation time
    plt.subplot(2, 2, 4)
    up_times = [metrics[c]['Time to upregulation'] for c in conc]
    up_times = [t if t != float('inf') else None for t in up_times]

    plt.plot(conc, up_times, 'go-')
    plt.axhline(y=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.xlabel('Antibiotic concentration')
    plt.ylabel('Time to upregulation')
    plt.title('Relationship between antibiotic concentration and upregulation time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_results/stress_response_analysis.png', dpi=300)
    plt.show()

    # Print detailed metrics
    print("\nStress response metrics comparison:")
    for c, m in metrics.items():
        print(f"\nAntibiotic concentration {c}:")
        for k, v in m.items():
            if v == float('inf'):
                print(f"  {k}: Infinite")
            else:
                print(f"  {k}: {v:.4f}")

    print("\nStress response mechanism analysis completed, results saved")

    return {'results': results, 'metrics': metrics}

def run_qsi_simulation(qsi_strength=1.0, qsi_timing=20.0, qsi_duration=None):

    print(f"Running QSI simulation (strength={qsi_strength}, timing={qsi_timing})")

    # Modify the solve_ivp function to support time-dependent parameters
    from scipy.integrate import solve_ivp

    # Create a new ODE function that includes time-dependent QSI effects
    def ode_system_with_qsi(t, y, base_params, qsi_params):
        # Copy base parameters
        params = copy.deepcopy(base_params)

        # Get QSI parameters
        qsi_strength, qsi_timing, qsi_duration = qsi_params

        # Apply QSI at specific time
        if qsi_timing <= t < (qsi_timing + qsi_duration if qsi_duration else float('inf')):
            # Modify relevant parameters to simulate QSI effects
            params.gamma = params.gamma * (1 + qsi_strength)  # Increase AHL degradation rate
            params.tau = params.tau * (1 + 0.5 * qsi_strength)  # Increase QS threshold
            params.sigma_0 = params.sigma_0 / (1 + 0.3 * qsi_strength)  # Reduce baseline AHL production rate
            params.sigma_S = params.sigma_S / (1 + 0.3 * qsi_strength)  # Reduce upregulated state AHL production rate

        # Use standard ODE system
        return ode_system(t, y, params)

    # Set parameters
    params = Parameters()

    # Set QSI parameters
    if qsi_duration is None:
        qsi_duration = params.T_antibiotic - qsi_timing if qsi_timing < params.T_antibiotic else float('inf')

    qsi_params = (qsi_strength, qsi_timing, qsi_duration)

    # Initial conditions
    I0 = 0.0  # Initial inactive biomass
    A0 = 0.05  # Initial downregulated biomass
    B0 = 0.0  # Initial upregulated biomass
    N0 = 1.0  # Initial nutrient concentration
    C0 = 0.0  # Initial antibiotic concentration
    S0 = 0.0  # Initial AHL concentration

    y0 = [I0, A0, B0, N0, C0, S0]

    # Simulation time
    t_span = (0, 60)
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Solve ODE system with QSI
    print("Solving ODE system with QSI...")
    solution = solve_ivp(
        lambda t, y: ode_system_with_qsi(t, y, params, qsi_params),
        t_span, y0, method='Radau', t_eval=t_eval, rtol=1e-6, atol=1e-9
    )

    if not solution.success:
        print("Failed to solve ODE system!")
        return None

    # Process results
    t = solution.t
    I, A, B, N, C, S = solution.y

    # Calculate total active biomass and other metrics
    total_active = A + B
    total_biomass = I + A + B

    # Safely calculate ratios, avoiding division by zero
    R = np.zeros_like(total_biomass)  # Active ratio
    Z = np.zeros_like(total_active)  # Downregulated ratio
    for i in range(len(total_biomass)):
        if total_biomass[i] > 1e-10:
            R[i] = total_active[i] / total_biomass[i]
        else:
            R[i] = 0

        if total_active[i] > 1e-10:
            Z[i] = A[i] / total_active[i]
        else:
            Z[i] = 1 if A[i] > 0 else 0

    # Calculate biofilm net growth rate
    BG = ((params.mu_A * N * A) / (params.K_N + N) +
          (params.mu_B * N * B) / (params.K_N + N) -
          params.k_A * A - params.k_B * B)

    # Create results dictionary
    results = {
        't': t,
        'I': I, 'A': A, 'B': B,
        'N': N, 'C': C, 'S': S,
        'total_active': total_active,
        'total_biomass': total_biomass,
        'Z': Z, 'R': R, 'BG': BG,
        'params': params,
        'qsi_params': qsi_params
    }

    # Plot results
    plt.figure(figsize=(15, 12))

    # 1. Biomass
    plt.subplot(2, 2, 1)
    plt.plot(t, I, 'k-', label='Inactive biomass (I)')
    plt.plot(t, A, 'b-', label='Downregulated biomass (A)')
    plt.plot(t, B, 'g-', label='Upregulated biomass (B)')
    plt.plot(t, total_active, 'r--', label='Total active biomass (A+B)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.axvline(x=qsi_timing, color='purple', linestyle='--', label='QSI addition')
    if qsi_duration != float('inf'):
        plt.axvline(x=qsi_timing + qsi_duration, color='purple', linestyle=':', label='QSI stop')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Biomass volume fraction')
    plt.title('Biomass changes over time')
    plt.legend()
    plt.grid(True)

    # 2. AHL concentration
    plt.subplot(2, 2, 2)
    plt.plot(t, S, 'r-', label='AHL concentration')
    plt.axhline(y=params.tau, color='gray', linestyle='--', label='Base QS threshold')
    # Show dynamic threshold under QSI influence
    if qsi_strength > 0:
        tau_with_qsi = params.tau * (1 + 0.5 * qsi_strength)
        qsi_idx_start = np.argmax(t >= qsi_timing)
        qsi_idx_end = np.argmax(t >= qsi_timing + qsi_duration) if qsi_duration != float('inf') else len(t)
        t_qsi = t[qsi_idx_start:qsi_idx_end]
        tau_qsi = np.ones_like(t_qsi) * tau_with_qsi
        plt.plot(t_qsi, tau_qsi, 'purple', linestyle='--', label='Threshold under QSI influence')

    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.axvline(x=qsi_timing, color='purple', linestyle='--', label='QSI addition')
    if qsi_duration != float('inf'):
        plt.axvline(x=qsi_timing + qsi_duration, color='purple', linestyle=':', label='QSI stop')
    plt.xlabel('Dimensionless time')
    plt.ylabel('AHL concentration [nM]')
    plt.title('AHL concentration changes over time')
    plt.legend()
    plt.grid(True)

    # 3. Downregulated ratio and active ratio
    plt.subplot(2, 2, 3)
    plt.plot(t, Z, 'b-', label='Downregulated biomass ratio (Z)')
    plt.plot(t, R, 'g-', label='Active biomass ratio (R)')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.axvline(x=qsi_timing, color='purple', linestyle='--', label='QSI addition')
    if qsi_duration != float('inf'):
        plt.axvline(x=qsi_timing + qsi_duration, color='purple', linestyle=':', label='QSI stop')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Ratio')
    plt.ylim(0, 1.1)
    plt.title('Downregulated biomass ratio and active biomass ratio')
    plt.legend()
    plt.grid(True)

    # 4. Nutrients, antibiotic and biofilm net growth rate
    plt.subplot(2, 2, 4)
    plt.plot(t, N, 'g-', label='Nutrient concentration')
    plt.plot(t, C, 'r-', label='Antibiotic concentration')
    plt.plot(t, BG, 'b--', label='Biofilm net growth rate')
    plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
    plt.axvline(x=qsi_timing, color='purple', linestyle='--', label='QSI addition')
    if qsi_duration != float('inf'):
        plt.axvline(x=qsi_timing + qsi_duration, color='purple', linestyle=':', label='QSI stop')
    plt.xlabel('Dimensionless time')
    plt.ylabel('Concentration/Growth rate')
    plt.title('Nutrient, antibiotic concentrations and biofilm net growth rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = f'simulation_results/qsi_simulation_s{qsi_strength}_t{qsi_timing}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("QSI simulation completed, results saved")

    return results