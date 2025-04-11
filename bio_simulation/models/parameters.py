class Parameters:
    def __init__(self):
        # Growth and decay parameters
        self.mu_A = 6.0  # Maximum growth rate of downregulated biomass [d^-1]
        self.mu_B = 4.0  # Maximum growth rate of upregulated biomass [d^-1]
        self.k_A = 0.1  # Natural decay rate of downregulated biomass [d^-1]
        self.k_B = 0.1  # Natural decay rate of upregulated biomass [d^-1]
        self.K_N = 0.2  # Half-saturation constant for nutrients [gm^-3]
        self.nu_A = 100.0  # Nutrient uptake rate for downregulated biomass [gm^-3 d^-1]
        self.nu_B = 100.0  # Nutrient uptake rate for upregulated biomass [gm^-3 d^-1]

        # Antibiotic-related parameters
        self.beta_A = 5.0  # Maximum inactivation rate of downregulated biomass by antibiotics [d^-1]
        self.beta_B = 0.5  # Maximum inactivation rate of upregulated biomass by antibiotics [d^-1]
        self.K_C = 0.2  # Half-saturation constant for antibiotics [gm^-3]
        self.K_C_prime = 0.2  # Half-saturation constant for AHL-induced antibiotic production [gm^-3]
        self.delta_A = 4.0  # Antibiotic consumption rate for downregulated biomass [gm^-3 d^-1]
        self.delta_B = 4.0  # Antibiotic consumption rate for upregulated biomass [gm^-3 d^-1]
        self.theta = 0.001  # Natural degradation rate of antibiotics [d^-1]
        self.n1 = 2.5  # Hill function exponent for antibiotics

        # Quorum sensing (QS)-related parameters
        self.omega = 4.0  # Conversion rate from downregulated to upregulated biomass [d^-1]
        self.psi = 1.0  # Conversion rate from upregulated to downregulated biomass [d^-1]
        self.tau = 10.0  # QS activation threshold [nM]
        self.sigma_0 = 30.0  # Baseline AHL production rate [nM d^-1]
        self.sigma_S = 300.0  # AHL production rate in upregulated state [nM d^-1]
        self.mu_S = 200.0  # AHL production rate induced by antibiotics [nM d^-1]
        self.gamma = 0.2  # Natural AHL degradation rate [d^-1]
        self.n2 = 2.5  # Hill function exponent for QS

        # Environmental conditions
        self.N_inf = 1.0  # External nutrient concentration [dimensionless]
        self.C_inf = 2.0  # External antibiotic concentration [gm^-3]

        # Antibiotic and QS inhibitor addition times
        self.T_antibiotic = 40.0  # Time when antibiotics are added [dimensionless]
        self.T_QSI = None  # Time when QS inhibitors are added [dimensionless], defaults to no addition

        # Diffusion-related parameters
        self.boundary_transfer = 0.2  # Transfer coefficient

        # Spatial model parameters
        self.spatial_simulation = False  # Whether to use spatial simulation
        self.num_layers = 10  # Number of layers in spatial simulation
