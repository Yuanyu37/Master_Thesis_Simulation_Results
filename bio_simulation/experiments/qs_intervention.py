import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import copy

# Ensure other modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.parameters import Parameters
from models.core_model import run_simulation, ode_system
from utils.visualization import plot_basic_results, plot_parameter_sensitivity

def run_qs_parameter_variation():
    print("Running quorum sensing parameter variation study...")

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
    print("Evaluating combined strategies...")

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
        "QSI simultaneous with antibiotics": {"color": "red", "linestyle": ":", "linewidth": 3, "marker": "^",
                                              "markevery": 30},
        "Antibiotics before QSI": {"color": "purple", "linestyle": "-", "linewidth": 2.5}
    }

    results = {}
    metrics = {}

    # Run reference simulation to get biofilm growth without any treatment
    reference_params = Parameters()
    reference_params.T_antibiotic = float('inf')  # Don't add antibiotics
    reference_params.add_QSI = False  # Don't add QSI
    initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0, QSI0]
    reference_results = run_simulation(reference_params, initial_conditions)

    # Get reference maximum biomass
    reference_max_biomass = np.max(reference_results['total_active'])
    print(f"Reference maximum biomass without treatment: {reference_max_biomass:.4f}")

    for i, (ab_time, ab_conc, qsi_time, qsi_strength) in enumerate(strategies):
        strategy_name = strategy_names[i]
        print(f"Evaluating strategy {i + 1}: {strategy_name}...")

        # Set parameters
        params = Parameters()
        params.T_antibiotic = ab_time
        params.C_inf = ab_conc

        # Set QSI parameters if needed
        if qsi_time is not None:
            params.add_QSI = True
            params.T_QSI = qsi_time
            params.QSI_inf = qsi_strength
        else:
            params.add_QSI = False

        # Set initial conditions with QSI
        initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0, QSI0]

        # Run simulation
        sim_results = run_simulation(params, initial_conditions)

        if sim_results is None:
            print(f"Simulation failed for strategy {strategy_name}!")
            continue

        # Save results
        results[strategy_name] = sim_results

        # Calculate performance metrics
        final_biomass = sim_results['total_active'][-1]

        # Calculate biofilm eradication rate - compare with reference maximum biomass instead of pre-treatment biomass
        eradication_rate = 1.0 - (final_biomass / reference_max_biomass)

        # Calculate strategy cost (simplified model)
        strategy_cost = 0.0
        if ab_conc > 0:
            strategy_cost += ab_conc * (sim_results['t'][-1] - ab_time)
        if qsi_strength > 0 and qsi_time is not None:
            strategy_cost += qsi_strength * (sim_results['t'][-1] - qsi_time) * 0.2  # Assume QSI cost is 20% of antibiotics

        # Compute metrics
        metrics[strategy_name] = {
            'Final active biomass': final_biomass,
            'Biofilm eradication rate': eradication_rate,
            'Max AHL concentration': np.max(sim_results['S']),
            'Strategy cost': strategy_cost,
            'Treatment duration': sim_results['t'][-1] - min(
                ab_time if ab_time is not None and ab_time < float('inf') else float('inf'),
                qsi_time if qsi_time is not None else float('inf'))
        }

        # Add QSI-related metrics if applicable
        if 'QSI' in sim_results and qsi_time is not None:
            metrics[strategy_name]['Max QSI concentration'] = np.max(sim_results['QSI'])

            # Calculate time below QS threshold after QSI addition
            if qsi_time is not None:
                qsi_idx = np.argmax(sim_results['t'] >= qsi_time)
                if qsi_idx > 0:
                    below_threshold_time = np.sum(sim_results['S'][qsi_idx:] < params.tau) * (
                            sim_results['t'][-1] - sim_results['t'][qsi_idx]) / len(sim_results['t'][qsi_idx:])
                    metrics[strategy_name]['Time below QS threshold'] = below_threshold_time

    # Compare different strategies
    if len(results) > 0:
        fig = plt.figure(figsize=(16, 15))  # Increased height for additional subplot

        # 1. Total active biomass
        ax1 = plt.subplot(3, 2, 1)

        # Draw lines in a specific order to ensure visibility: draw main lines first, then those likely to be covered
        plot_order = ["Antibiotics before QSI", "QSI before antibiotics", "Antibiotics only",
                      "QSI simultaneous with antibiotics", "QSI only"]

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
        ax2 = plt.subplot(3, 2, 2)

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
        ax3 = plt.subplot(3, 2, 3)

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

        # 4. QSI concentration (new plot)
        ax4 = plt.subplot(3, 2, 4)

        for name in plot_order:
            if name in results and 'QSI' in results[name]:
                result = results[name]
                style = line_styles[name]
                ax4.plot(result['t'], result['QSI'], label=name, **style)

        # Mark treatment time points
        for name, result in results.items():
            idx = strategies[list(strategy_names).index(name)]
            if idx[0] is not None:
                ax4.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.3)
            if idx[2] is not None:
                ax4.axvline(x=idx[2], color='purple', linestyle='--', alpha=0.3)

        ax4.set_xlabel('Dimensionless time')
        ax4.set_ylabel('QSI concentration')
        ax4.set_title('QSI concentration under different strategies')
        ax4.legend(loc='best')
        ax4.grid(True)

        # 5. Performance metrics as bar charts with dual Y axes
        ax5 = plt.subplot(3, 2, 5)

        # Ensure strategy names and data values correspond correctly
        strategy_names_sorted = list(metrics.keys())
        x = np.arange(len(strategy_names_sorted))
        width = 0.35

        # Ensure data is extracted by name
        eradication_rates = [metrics[name]['Biofilm eradication rate'] for name in strategy_names_sorted]
        final_biomass = [metrics[name]['Final active biomass'] for name in strategy_names_sorted]

        for i, rate in enumerate(eradication_rates):
            if rate > 1.0:  # Eradication rate should not exceed 1
                eradication_rates[i] = 1.0
            elif rate < 0.0:  # Eradication rate should not be less than 0
                eradication_rates[i] = 0.0

        bars1 = ax5.bar(x - width / 2, eradication_rates, width, color='blue', alpha=0.7,
                        label='Biofilm eradication rate')

        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x + width / 2, final_biomass, width, color='orange', alpha=0.7,
                             label='Final active biomass')

        ax5.set_ylim(0, 1.05)
        max_biomass = max(final_biomass) if max(final_biomass) > 0 else 0.01
        ax5_twin.set_ylim(0, max_biomass * 1.2)

        # Add grid lines to improve readability
        ax5.grid(True, alpha=0.3)

        # Add labels and legend
        ax5.set_xlabel('Strategy')
        ax5.set_ylabel('Biofilm eradication rate', color='blue')
        ax5_twin.set_ylabel('Final active biomass', color='orange')
        ax5.set_title('Comparison of strategy effectiveness')
        ax5.set_xticks(x)
        ax5.set_xticklabels(strategy_names_sorted, rotation=45, ha='right')

        # Merge legends from both axes
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax5.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9)

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax5_twin.annotate(f'{height:.4f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom',
                              fontsize=9)

        # 6. Upregulation ratio comparison
        ax6 = plt.subplot(3, 2, 6)

        for name in plot_order:
            if name in results:
                result = results[name]
                style = line_styles[name]
                # Calculate upregulation ratio (1-Z)
                upregulation_ratio = 1 - result['Z']
                ax6.plot(result['t'], upregulation_ratio, label=name, **style)

        # Mark treatment time points
        for name, result in results.items():
            idx = strategies[list(strategy_names).index(name)]
            if idx[0] is not None:
                ax6.axvline(x=idx[0], color='gray', linestyle='--', alpha=0.3)
            if idx[2] is not None:
                ax6.axvline(x=idx[2], color='purple', linestyle='--', alpha=0.3)

        ax6.set_xlabel('Dimensionless time')
        ax6.set_ylabel('Upregulation ratio')
        ax6.set_title('Upregulation ratio under different strategies')
        ax6.set_ylim(0, 1.1)
        ax6.legend(loc='best')
        ax6.grid(True)

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
        params = Parameters()
        params.C_inf = c
        params.T_antibiotic = 20.0
        params.add_QSI = False

        # Run simulation with QSI state variable
        initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0, QSI0]
        sim_results = run_simulation(params, initial_conditions)

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
            time_to_upregulation = float('inf')

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

def run_qsi_state_variable_experiment():
    print("Running QSI as state variable experiment...")

    qsi_concentrations = [0.0, 0.5, 1.0, 2.0, 5.0]

    results = {}

    for qsi_conc in qsi_concentrations:
        params = Parameters()
        params.add_QSI = True
        params.T_QSI = 20.0
        params.QSI_inf = qsi_conc

        # Run simulation
        initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]
        sim_results = run_simulation(params, initial_conditions)

        if sim_results is None:
            print(f"Simulation failed, QSI concentration = {qsi_conc}!")
            continue

        results[qsi_conc] = sim_results

        save_path = f'simulation_results/qsi_state_var_conc{qsi_conc}.png'
        plot_basic_results(sim_results, params, save_path)

    # Compare results for different QSI concentrations
    if len(results) > 0:
        plt.figure(figsize=(15, 10))

        # 1. Compare total active biomass
        plt.subplot(2, 2, 1)
        for conc, result in results.items():
            plt.plot(result['t'], result['total_active'], label=f'QSI={conc}')
        plt.axvline(x=params.T_QSI, color='purple', linestyle='--', label='QSI addition')
        plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
        plt.xlabel('Dimensionless time')
        plt.ylabel('Total active biomass')
        plt.title('Total active biomass for different QSI concentrations')
        plt.legend()
        plt.grid(True)

        # 2. Compare AHL concentration
        plt.subplot(2, 2, 2)
        for conc, result in results.items():
            plt.plot(result['t'], result['S'], label=f'QSI={conc}')
        plt.axhline(y=params.tau, color='gray', linestyle='--', label='QS threshold')
        plt.axvline(x=params.T_QSI, color='purple', linestyle='--', label='QSI addition')
        plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
        plt.xlabel('Dimensionless time')
        plt.ylabel('AHL concentration [nM]')
        plt.title('AHL concentration for different QSI concentrations')
        plt.legend()
        plt.grid(True)

        # 3. Compare QSI concentration
        plt.subplot(2, 2, 3)
        for conc, result in results.items():
            plt.plot(result['t'], result['QSI'], label=f'QSI={conc}')
        plt.axvline(x=params.T_QSI, color='purple', linestyle='--', label='QSI addition')
        plt.xlabel('Dimensionless time')
        plt.ylabel('QSI concentration')
        plt.title('QSI concentration over time')
        plt.legend()
        plt.grid(True)

        # 4. Compare upregulation ratio
        plt.subplot(2, 2, 4)
        for conc, result in results.items():
            plt.plot(result['t'], 1 - result['Z'], label=f'QSI={conc}')  # 1-Z = upregulation ratio
        plt.axvline(x=params.T_QSI, color='purple', linestyle='--', label='QSI addition')
        plt.axvline(x=params.T_antibiotic, color='gray', linestyle='--', label='Antibiotic addition')
        plt.xlabel('Dimensionless time')
        plt.ylabel('Upregulation ratio')
        plt.title('Upregulation ratio for different QSI concentrations')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('simulation_results/qsi_state_var_comparison.png', dpi=300)
        plt.show()

    print("QSI as state variable experiment completed, results saved")

    return results