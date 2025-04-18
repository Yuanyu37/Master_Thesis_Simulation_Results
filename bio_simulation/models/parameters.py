class Parameters:
    def __init__(self):
        # Growth and decay parameters
        self.mu_A = 6.0  # Maximum growth rate of down-regulated biomass [d^-1]
        self.mu_B = 4.0  # Maximum growth rate of up-regulated biomass [d^-1]
        self.k_A = 0.1  # Natural decay rate of down-regulated biomass [d^-1]
        self.k_B = 0.1  # Natural decay rate of up-regulated biomass [d^-1]
        self.K_N = 0.2  # Half-saturation constant for nutrient [gm^-3]
        self.nu_A = 100.0  # Nutrient uptake rate [gm^-3 d^-1]
        self.nu_B = 100.0  # Nutrient uptake rate [gm^-3 d^-1]

        # Antibiotic-related parameters
        self.beta_A = 5.0  # Maximum rate at which antibiotics inactivate down-regulated biomass [d^-1]
        self.beta_B = 0.5  # Maximum rate at which antibiotics inactivate up-regulated biomass [d^-1]
        self.K_C = 0.2  # Half-saturation constant for antibiotic [gm^-3]
        self.K_C_prime = 0.2  # Half-saturation constant for antibiotic in AHL production [gm^-3]
        self.delta_A = 4.0  # Antibiotic consumption rate [gm^-3 d^-1]
        self.delta_B = 4.0  # Antibiotic consumption rate [gm^-3 d^-1]
        self.theta = 0.001  # Natural degradation rate of antibiotic [d^-1]
        self.n1 = 2.5  # Hill function exponent for antibiotic

        # Quorum sensing related parameters
        self.omega = 4.0  # Conversion rate from down-regulated to up-regulated [d^-1]
        self.psi = 1.0  # Conversion rate from up-regulated to down-regulated [d^-1]
        self.tau = 10.0  # QS activation threshold [nM]
        self.sigma_0 = 30.0  # Basal AHL production rate [nM d^-1]
        self.sigma_S = 300.0  # AHL production rate in up-regulated state [nM d^-1]
        self.mu_S = 200.0  # Antibiotic-induced AHL production rate [nM d^-1]
        self.gamma = 0.2  # Natural degradation rate of AHL [d^-1]
        self.n2 = 2.5  # Hill function exponent for QS

        # Environmental conditions
        self.N_inf = 1.0  # External nutrient concentration [dimensionless]
        self.C_inf = 2.0  # External antibiotic concentration [gm^-3]

        # Antibiotic addition time
        self.T_antibiotic = 40.0  # Time at which antibiotic is added [dimensionless]

        # Diffusion-related parameters
        self.boundary_transfer = 0.2  # Transfer coefficient

        # QSI (Quorum Sensing Inhibitor)
        self.QSI_decay = 0.1  # Natural degradation rate of QSI [d^-1]
        self.QSI_inf = 0.0  # External QSI concentration [units depend on your model]
        self.QSI_inhibition_constant = 0.5  # Half-inhibition constant of QSI on AHL production
        self.QSI_hill_exponent = 1.0  # Hill function exponent for QSI inhibition
        self.add_QSI = False  # Whether to add QSI
        self.T_QSI = float('inf')  # Time at which QSI is added