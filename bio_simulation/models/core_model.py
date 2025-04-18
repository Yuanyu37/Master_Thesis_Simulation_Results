import numpy as np
from scipy.integrate import solve_ivp

def ode_system(t, y, params):
    if len(y) == 6:
        I, A, B, N, C, S = y
        QSI = 0.0
    else:
        I, A, B, N, C, S, QSI = y

    # Prevent negative values for variables
    I = max(0, I)
    A = max(0, A)
    B = max(0, B)
    N = max(0, N)
    C = max(0, C)
    S = max(0, S)
    QSI = max(0, QSI)

    # Calculate total biomass
    M = I + A + B

    # Antibiotic control - add at specific time point
    C_external = 0.0
    if t >= params.T_antibiotic:
        C_external = params.C_inf

    # QSI control
    QSI_external = 0.0
    if params.add_QSI and t >= params.T_QSI:
        QSI_external = params.QSI_inf

    # Calculate Hill functions
    if C == 0:
        Hill_C_A = 0
        Hill_C_B = 0
    else:
        Hill_C_A = (C ** params.n1) / (params.K_C ** params.n1 + C ** params.n1)
        Hill_C_B = (C ** params.n1) / (params.K_C ** params.n1 + C ** params.n1)

    if S == 0:
        Hill_S_up = 0
        Hill_S_down = 1
    else:
        Hill_S_up = (S ** params.n2) / (params.tau ** params.n2 + S ** params.n2)
        Hill_S_down = (params.tau ** params.n2) / (params.tau ** params.n2 + S ** params.n2)

    # Calculate QSI inhibition factor (range 0-1, 0 means complete inhibition, 1 means no inhibition)
    if QSI == 0:
        QSI_inhibition = 1.0  # No inhibition
    else:
        QSI_inhibition = params.QSI_inhibition_constant ** params.QSI_hill_exponent / \
                         (params.QSI_inhibition_constant ** params.QSI_hill_exponent +
                          QSI ** params.QSI_hill_exponent)

    # 1. Inert biomass equation
    dI_dt = params.beta_A * Hill_C_A * A + params.beta_B * Hill_C_B * B

    # 2. Down-regulated biomass equation
    growth_A = (params.mu_A * N * A) / (params.K_N + N)
    dA_dt = growth_A - params.beta_A * Hill_C_A * A + \
            params.psi * Hill_S_down * B - params.omega * Hill_S_up * A - params.k_A * A

    # 3. Up-regulated biomass equation
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

    # 6. AHL signal molecule equation
    base_production = params.sigma_0 * (A + B) * QSI_inhibition  # QSI inhibits base production
    stress_production = params.mu_S * (A + B) * C / (params.K_C_prime + C) * QSI_inhibition  # QSI inhibits stress production
    up_production = params.sigma_S * Hill_S_up * B * QSI_inhibition  # QSI inhibits up-regulated state production

    dS_dt = base_production + stress_production + up_production - params.gamma * S - params.boundary_transfer * S

    # 7. QSI equation
    dQSI_dt = params.boundary_transfer * (QSI_external - QSI) - params.QSI_decay * QSI

    # If original input does not include QSI, return 6 values
    if len(y) == 6:
        return [dI_dt, dA_dt, dB_dt, dN_dt, dC_dt, dS_dt]
    else:
        return [dI_dt, dA_dt, dB_dt, dN_dt, dC_dt, dS_dt, dQSI_dt]

def ode_system_enhanced(t, y, params):
    # Initially based on the original ODE system
    derivatives = ode_system(t, y, params)

    # Unpack variables based on input length
    if len(y) == 6:
        I, A, B, N, C, S = y
    else:
        I, A, B, N, C, S, QSI = y

    if not hasattr(ode_system_enhanced, "qsi_activated"):
        ode_system_enhanced.qsi_activated = False

    if S >= params.tau and not ode_system_enhanced.qsi_activated:
        ode_system_enhanced.qsi_activated = True
        print(f"QS activation time: {t:.2f}")

    return derivatives

def run_simulation(params, initial_conditions=None, t_span=(0, 60), t_points=500):
    if initial_conditions is None:
        if hasattr(params, 'add_QSI') and params.add_QSI:
            initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0]
        else:
            initial_conditions = [0.0, 0.05, 0.0, 1.0, 0.0, 0.0]
    else:
        if hasattr(params, 'add_QSI') and params.add_QSI and len(initial_conditions) == 6:
            initial_conditions = initial_conditions + [0.0]

    # Set time points
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Solve ODE system
    print("Solving ODE system...")
    solution = solve_ivp(ode_system_enhanced, t_span, initial_conditions,
                         args=(params,), method='Radau',
                         t_eval=t_eval, rtol=1e-6, atol=1e-9)
    if not solution.success:
        print("Failed to solve ODE system!")
        return None

    t = solution.t

    if len(initial_conditions) == 7:
        I, A, B, N, C, S, QSI = solution.y
    else:
        I, A, B, N, C, S = solution.y
        QSI = np.zeros_like(t)

    # Calculate total active biomass and other metrics
    total_active = A + B
    total_biomass = I + A + B

    # Safely calculate ratios, avoiding division by zero
    R = np.zeros_like(total_biomass)  # Active ratio
    Z = np.zeros_like(total_active)  # Down-regulated ratio

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

    results = {
        't': t,
        'I': I, 'A': A, 'B': B,
        'N': N, 'C': C, 'S': S,
        'QSI': QSI,
        'total_active': total_active,
        'total_biomass': total_biomass,
        'Z': Z, 'R': R, 'BG': BG
    }
    return results