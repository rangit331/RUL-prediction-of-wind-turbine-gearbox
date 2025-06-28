import numpy as np
import matplotlib.pyplot as plt
class GearWearModel:
    def __init__(self):
        # Gear parameters
        self.z_s = 19
        self.z_p = 31
        self.z_r = 81
        self.m = 3.2e-3
        self.alpha_0 = np.deg2rad(20)
        self.B = 0.0381
        self.rho = 7850 #density assumed
        self.T = 2825
        self.n_s = 1200
        self.omega_s = 2 * np.pi * self.n_s / 60
        self.E = 2.1e11
        self.nu = 0.3
        self.failure_threshold = 28.71e-3
        self.r_b1 = 0.0283

        # Derived parameters
        self.r_p1 = self.m * self.z_s / 2
        self.r_p2 = self.m * self.z_p / 2
        self.tooth_height = 2.25 * self.m
        self.meshes_per_cycle = 3.2
        self.F = self.T / self.r_p1
        self.y = 0

        # Bayesian parameters
        self.k_prior_mean = 1.27e-15
        self.k_prior_std = 1e-15
        self.sigma = 0.1e-3  # Increased for stability
        self.k_mean = self.k_prior_mean
        self.k_std = self.k_prior_std
        self.h_initial = 0

        # Experimental data
        self.runs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.cycles = [89600, 268800, 358400, 448000, 582400, 716800, 868000, 1100000, 1136800, 1271200]
        self.mass_loss_obs = [1.4e-3, 4.45e-3, 8.12e-3, 10.9e-3, 13.81e-3, 17e-3, 20.13e-3, 23.14e-3, 26.04e-3, 28.71e-3]

        # Cache
        self._cached_y = None
        self._cached_p = None
        self._cached_a_H = None
        self._cached_s = None

    def safe_sqrt(self, x):
        return np.sqrt(np.maximum(x, 0))

    def calculate_contact_pressure(self):
        if self._cached_y == self.y and self._cached_p is not None:
            return self._cached_p, self._cached_a_H
        R_1 = self.r_b1 * np.tan(self.alpha_0) + self.y
        R_2 = max(self.r_b1 * np.tan(self.alpha_0) - self.y, 1e-6)
        R_star = 1 / (1 / R_1 + 1 / R_2)
        E_star = self.E / (2 * (1 - self.nu ** 2))
        arg = 3 * self.F * R_star / (2 * np.pi * self.B * E_star)
        if arg < 0:
            print("Warning: Negative argument in sqrt for a_H")
            return 0, 0
        self.a_H = self.safe_sqrt(arg)
        self.p_N = 3 * self.F / (2 * np.pi * self.a_H * self.B) if self.a_H > 0 else 0
        self._cached_y = self.y
        self._cached_p = self.p_N
        self._cached_a_H = self.a_H
        return self.p_N, self.a_H

    def calculate_sliding_distance(self):
        if self._cached_y == self.y and self._cached_s is not None:
            return self._cached_s
        R_1 = self.safe_sqrt((self.r_p1 * np.cos(self.alpha_0))**2 + (self.r_p2 * np.sin(self.alpha_0) - self.y)**2)
        y_1e = self.safe_sqrt(max(R_1 - (self.r_p1 * np.cos(self.alpha_0) + self.a_H)**2, 0)) - self.r_p1 * np.sin(self.alpha_0)
        R_2 = self.safe_sqrt((self.r_p2 * np.cos(self.alpha_0) - self.a_H)**2 + (self.r_p2 * np.sin(self.alpha_0) - self.y)**2)
        y_1d = self.safe_sqrt(max(R_1 - (self.r_p1 * np.cos(self.alpha_0) - self.a_H)**2, 0)) - self.r_p1 * np.sin(self.alpha_0)
        term1 = self.a_H
        term2 = self.safe_sqrt(max(R_2**2 - (self.r_p2 * np.sin(self.alpha_0) - y_1d), 0))
        term3 = self.r_p2 * np.cos(self.alpha_0)
        s = abs(term1 - term2 + term3) * 0.05  # Adjusted scaling factor
        self._cached_s = s
        self._cached_y = self.y
        return s

    def calculate_wear_depth(self, h_prev, k, p, s):
        h = h_prev + k * p * s
        return max(min(h, self.tooth_height), 0)

    def calculate_mass_loss(self, h_s):
        area_loss = h_s * self.tooth_height / 2
        volume_loss = area_loss * self.B
        mass_loss_s = volume_loss * self.rho
        total_mass_loss = mass_loss_s * self.z_s
        return max(total_mass_loss, 0)

    def bayesian_update(self, mass_loss_obs, current_cycle):
        k_range = np.linspace(1e-16, 5e-15, 1000)
        log_posterior = np.zeros_like(k_range)
        
        p, _ = self.calculate_contact_pressure()
        s = self.calculate_sliding_distance()
        num_meshes = current_cycle * self.meshes_per_cycle
        
        for i, k in enumerate(k_range):
            h_per_mesh = self.calculate_wear_depth(0, k, p, s)
            m_mod = self.calculate_mass_loss(h_per_mesh) * num_meshes
            log_likelihood = -0.5 * ((mass_loss_obs - m_mod) ** 2 / self.sigma ** 2) - np.log(np.sqrt(2 * np.pi) * self.sigma)
            log_prior = -0.5 * ((k - self.k_mean) ** 2 / max(self.k_std ** 2, 1e-30)) - np.log(np.sqrt(2 * np.pi) * max(self.k_std, 1e-18))
            log_posterior[i] = np.clip(log_likelihood + log_prior, -1e10, 1e10)
        
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        if np.all(posterior == 0):
            print("Warning: Posterior is all zeros, returning prior mean")
            return self.k_mean, self.k_std
        posterior /= np.trapz(posterior, k_range)
        
        k_mean_new = np.trapz(k_range * posterior, k_range)
        k_std_new = np.sqrt(np.trapz((k_range - k_mean_new) ** 2 * posterior, k_range))
        self.k_mean = np.clip(k_mean_new, 1e-16, 5e-15)
        self.k_std = max(k_std_new, 1e-18)
        print(f"Bayesian Update: k_mean={self.k_mean:.2e}, k_std={self.k_std:.2e}")
        return self.k_mean, self.k_std

    def predict_rul(self, current_cycle, h_current, inspection_idx):
        h = h_current
        num_meshes = current_cycle * self.meshes_per_cycle
        total_mass = self.calculate_mass_loss(h / num_meshes) * num_meshes
        cycle_count = 0
        max_cycles = int(5e6)
        p, _ = self.calculate_contact_pressure()
        s = self.calculate_sliding_distance()
        while total_mass < self.failure_threshold and cycle_count < max_cycles:
            h = self.calculate_wear_depth(h, self.k_mean, p, s)
            mass_loss_cycle = self.calculate_mass_loss(h / ((cycle_count + current_cycle) * self.meshes_per_cycle)) * self.meshes_per_cycle
            total_mass += mass_loss_cycle
            cycle_count += 1
            if cycle_count % 100000 == 0:
                print(f"Run #{inspection_idx+5}, Cycle {cycle_count}: h={h:.2e}, mass_loss={mass_loss_cycle:.2e}, total_mass={total_mass:.2e}")
        if cycle_count >= max_cycles:
            print(f"Max cycles reached at Run #{inspection_idx+5}")
            return np.inf
        return cycle_count

    def run_prognostics(self):
        print("RUL predictions at each inspection point:\n")
        for idx in range(len(self.runs)):
            current_cycle = self.cycles[idx]
            m_obs = self.mass_loss_obs[idx]
            
            p, _ = self.calculate_contact_pressure()
            s = self.calculate_sliding_distance()
            num_meshes = current_cycle * self.meshes_per_cycle
            h_per_mesh = self.calculate_wear_depth(0, self.k_mean, p, s)
            h = h_per_mesh * num_meshes
            self.y = h
            m_mod = self.calculate_mass_loss(h_per_mesh) * num_meshes
            print(f"Run #{self.runs[idx]}: m_obs={m_obs:.2e}, m_mod={m_mod:.2e}, k_mean={self.k_mean:.2e}")
            
            self.bayesian_update(m_obs, current_cycle)
            
            rul = self.predict_rul(current_cycle, h, idx)
            actual_rul = self.cycles[-1] - current_cycle
            print(f"Run #{self.runs[idx]} (Cycle: {current_cycle}): 🔮 RUL = {rul:.2e} cycles, Actual RUL = {actual_rul:.2e} cycles\n")
            
            self.h_initial = h
            
    # Data from the output
runs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
cycles = [89600, 268800, 358400, 448000, 582400, 716800, 868000, 1100000, 1136800, 1271200]
m_obs = [1.40, 4.45, 8.12, 10.90, 13.81, 17.00, 20.13, 23.14, 26.04, 28.71]  # in grams
m_mod = [2.90, 4.21, 5.93, 10.10, 14.20, 17.00, 20.60, 25.50, 23.90, 29.10]  # in grams
rul_pred = [3.43e6, 3.32e6, 2.26e6, 1.26e6, 8.06e5, 6.10e5, 3.90e5, 1.44e5, 2.43e5, 0]  # in cycles
rul_actual = [1.18e6, 1.00e6, 9.13e5, 8.23e5, 6.89e5, 5.54e5, 4.03e5, 1.71e5, 1.34e5, 0]  # in cycles

# Plot 1: Mass Loss (Observed vs. Predicted)
plt.figure(figsize=(10, 6))
plt.plot(cycles, m_obs, 'bo-', label='Observed Mass Loss (m_obs)', markersize=8)
plt.plot(cycles, m_mod, 'r^-', label='Predicted Mass Loss (m_mod)', markersize=8)
plt.xlabel('Cycles', fontsize=12)
plt.ylabel('Mass Loss (grams)', fontsize=12)
plt.title('Mass Loss vs. Cycles', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('mass_loss_plot.png')  # Save the plot
plt.show()

# Plot 2: RUL (Predicted vs. Actual)
plt.figure(figsize=(10, 6))
plt.plot(cycles, rul_pred, 'go-', label='Predicted RUL', markersize=8)
plt.plot(cycles, rul_actual, 'm^-', label='Actual RUL', markersize=8)
plt.xlabel('Cycles', fontsize=12)
plt.ylabel('RUL (Cycles)', fontsize=12)
plt.title('Remaining Useful Life (RUL) vs. Cycles', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('rul_plot.png')  # Save the plot
plt.show()

if __name__ == "__main__":
    model = GearWearModel()
    model.run_prognostics()
