import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    if not os.path.exists('simulation_results'):
        os.makedirs('simulation_results')

    print("Starting biofilm quorum sensing and antibiotic simulation system...")

    # Importing modules - placed here to avoid circular import issues
    from models.parameters import Parameters
    from models.core_model import run_simulation
    from experiments.basic_simulation import run_experiment1_final, run_timing_experiment
    from experiments.qs_intervention import (
        run_qs_parameter_variation,
        run_qsi_simulation,
        evaluate_combined_strategies,
        analyze_stress_response
    )

    # Main menu loop
    while True:
        print("\n===== Biofilm - Quorum Sensing Model Simulation System =====")
        print("1. Run basic explanatory simulation")
        print("2. Run antibiotic addition timing experiment")
        print("3. Run quorum sensing parameter variation analysis")
        print("4. Run quorum sensing inhibitor simulation")
        print("5. Evaluate combined strategies")
        print("6. Analyze stress response mechanism")
        print("0. Exit")

        choice = input("\nPlease select the experiment to run (0-6): ")

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
            # Get user input for QSI parameters
            try:
                qsi_strength = float(input("Enter QSI strength (default=1.0): ") or "1.0")
                qsi_timing = float(input("Enter QSI addition time (default=20.0): ") or "20.0")
                qsi_duration_input = input("Enter QSI duration (leave blank for infinite): ")
                qsi_duration = float(qsi_duration_input) if qsi_duration_input else None

                results = run_qsi_simulation(qsi_strength, qsi_timing, qsi_duration)
            except ValueError:
                print("Invalid input, please enter a number")
                continue
        elif choice == '5':
            results = evaluate_combined_strategies()
        elif choice == '6':
            results = analyze_stress_response()
        else:
            print("Invalid choice, please enter again")
            continue

        print("\nExperiment completed! Please check the results in the 'simulation_results' directory.")
        input("Press Enter to continue...")

if __name__ == "__main__":
    main()
