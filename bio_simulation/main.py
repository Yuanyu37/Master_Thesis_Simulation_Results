import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')

    print("Starting biofilm quorum sensing and antibiotic simulation system...")

    # Import modules - placed here to avoid circular import issues
    from models.parameters import Parameters
    from models.core_model import run_simulation
    from experiments.basic_simulation import run_experiment1_final, run_timing_experiment
    from experiments.qs_intervention import (
        run_qs_parameter_variation,
        run_qsi_state_variable_experiment,
        evaluate_combined_strategies,
        analyze_stress_response,
    )

    # Main menu loop
    while True:
        print("\n===== Biofilm-Quorum Sensing Model Simulation System =====")
        print("1. Run basic descriptive simulation")
        print("2. Run antibiotic addition timing experiment")
        print("3. Run quorum sensing parameter sensitivity analysis")
        print("4. Run QSI as state variable simulation")
        print("5. Evaluate combined strategies")
        print("6. Analyze stress response mechanisms")
        print("0. Exit")

        choice = input("\nPlease select an experiment to run (0-6): ")

        if choice == '0':
            print("Exiting program")
            break
        elif choice == '1':
            results = run_experiment1_final()
        elif choice == '2':
            results = run_timing_experiment()
        elif choice == '3':
            results = run_qs_parameter_variation()
        elif choice == '4':
            results = run_qsi_state_variable_experiment()
        elif choice == '5':
            results = evaluate_combined_strategies()
        elif choice == '6':
            results = analyze_stress_response()
        else:
            print("Invalid selection, please try again")
            continue

        print("\nExperiment completed! Please check the simulation_results directory for results.")
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()