import numpy as np
from scipy.integrate import solve_ivp

def ode_system(t, y, params):
    # Unpack variables
    I, A, B, N, C, S = y

    # Prevent variables from becoming negative
    I = max(0, I)
    A = max(0, A)
    B = max(0, B)
    N = max(0, N)
    C = max(0, C)
    S = max(0, S)

    # Calculate total biomass
    M = I + A + B

    # Antibiotic control - added at specific time points
    C_external = 0.0
    if t >= params.T_antibiotic:
        C_external = params.C_inf

    # QS inhibitor control
    gamma = params.gamma
    tau = params.tau
    if params.T_QSI is not None and t >= params.T_QSI:
        gamma = gamma * 10  # Increase AHL degradation rate by 10 times

    # Calculate Hill function, adding safety checks to prevent numerical issues
    if C == 0:
        Hill_C_A = 0
        Hill_C_B = 0
    else:
        Hill_C_A = (C ** params.n1) / (params.K_C ** params.n1 + C ** params.n1)
        Hill_C_B = (C ** params.n1) / (params.K_C ** params.n1 + C ** params.n1)

    if S == 0:
        Hill_S_up = 0
        Hill_S_down = 1  # When S = 0, downregulation is complete
    else:
        Hill_S_up = (S ** params.n2) / (tau ** params.n2 + S ** params.n2)
        Hill_S_down = (tau ** params.n2) / (tau ** params.n2 + S ** params.n2)

    # 1. Inert biomass equation
    dI_dt = params.beta_A * Hill_C_A * A + params.beta_B * Hill_C_B * B

    # 2. Downregulated biomass equation
    growth_A = (params.mu_A * N * A) / (params.K_N + N)
    dA_dt = growth_A - params.beta_A * Hill_C_A * A + \
            params.psi * Hill_S_down * B - params.omega * Hill_S_up * A - params.k_A * A

    # 3. Upregulated biomass equation
    growth_B = (params.mu_B * N * B) / (params.K_N + N)
    dB_dt = growth_B - params.beta_B * Hill_C_B * B - \
            params.psi * Hill_S_down * B + params.omega * Hill_S_up * A - params.k_B * B

    # 4. Nutrient equation
    consumption_A = (params.nu_A * N * A) / (params.K_N + N)
    consumption_B = (params.nu_B * N * B) / (params.K_N + N)
    dN_dt = params.boundary_transfer * (params.N_inf - N) - consumption_A - consumption_B

    # 5. Antibiotic equation
    antibiotic_consumption = params.delta_A * Hill_C_A * A + params.delta_B * Hill_C_B * B
    dC_dt = params.boundary_transfer * (C_external - C) - antibiotic_consumption - params.theta * C

    # 6. AHL signaling molecule equation
    base_production = params.sigma_0 * (A + B)
    stress_production = params.mu_S * (A + B) * C / (params.K_C_prime + C)
    up_production = params.sigma_S * Hill_S_up * B

    dS_dt = base_production + stress_production + up_production - gamma * S - params.boundary_transfer * S

    return [dI_dt, dA_dt, dB_dt, dN_dt, dC_dt, dS_dt]

def ode_system_enhanced(t, y, params):
    # Initially based on the original ODE system
    derivatives = ode_system(t, y, params)

    # Add event detection
    I, A, B, N, C, S = y
    if not hasattr(ode_system_enhanced, "qsi_activated"):
        ode_system_enhanced.qsi_activated = False

    if S >= params.tau and not ode_system_enhanced.qsi_activated:
        ode_system_enhanced.qsi_activated = True
        print(f"QS activation time: {t:.2f}")

    return derivatives

def run_simulation(params, initial_conditions=None, t_span=(0, 60), t_points=500):
    if initial_conditions is None:
        initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0]  # [I0, A0, B0, N0, C0, S0]

    # Set time points
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Solve the ODE system
    print("Solving ODE system...")
    solution = solve_ivp(ode_system_enhanced, t_span, initial_conditions,
                         args=(params,), method='Radau',
                         t_eval=t_eval, rtol=1e-6, atol=1e-9)

    if not solution.success:
        print("Failed to solve ODE system!")
        return None

    # Process results
    t = solution.t
    I, A, B, N, C, S = solution.y

    # Calculate total active biomass and other indicators
    total_active = A + B
    total_biomass = I + A + B

    # Safely calculate ratios to avoid division by zero
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

    return {
        't': t,
        'I': I, 'A': A, 'B': B,
        'N': N, 'C': C, 'S': S,
        'total_active': total_active,
        'total_biomass': total_biomass,
        'Z': Z, 'R': R, 'BG': BG
    }
