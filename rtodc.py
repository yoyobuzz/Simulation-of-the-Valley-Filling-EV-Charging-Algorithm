import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import Dict, List

class RealTimeOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,  # Total scheduling horizon
        D: np.ndarray,  # Base load profile
        beta: float,  # Lipschitz constant
        K: int,  # Number of iterations in each time slot
    ):
        self.T = T
        self.D = D
        self.beta = beta
        self.K = K
        self.gamma = 0.5 / beta  # 0 < gamma < 1/β
        self.t = 0  # Current time slot
        self.EVs: Dict[int, Dict] = {}  # EVs data
        self.Nt: List[int] = []  # Active EVs at time t
        self.total_load = np.zeros(self.T)
        self.convergence_history = []
        self.ev_id_list = []  # List of EV IDs

    def add_EV(
        self,
        ev_id: int,
        arrival_time: int,
        deadline: int,
        energy_needed: float,
        max_charging_rate: float,
    ):
        """Add an EV to the system."""
        ev_data = {
            'id': ev_id,
            'arrival': arrival_time,
            'deadline': deadline,
            'R_t': energy_needed,  # Remaining energy to charge
            'max_rate': max_charging_rate,
            'active': False,
            'r_nk': None,  # Charging profile at iteration k
            'r_n_history': [],  # Charging rate over time slots
        }
        self.EVs[ev_id] = ev_data
        self.ev_id_list.append(ev_id)

    def run(self):
        """Run the real-time optimal decentralized charging algorithm."""
        for t in range(self.T):
            self.t = t
            # Step 1: Update active EVs N_t
            Nt = []
            for ev_id, ev_data in self.EVs.items():
                if ev_data['arrival'] <= t < ev_data['deadline'] and ev_data['R_t'] > 1e-6:
                    Nt.append(ev_id)
                    ev_data['active'] = True
                else:
                    ev_data['active'] = False
            self.Nt = Nt

            if not Nt:
                # No active EVs at time t
                self.total_load[t] = self.D[t]
                self.convergence_history.append(np.var(self.total_load[:t+1]))
                continue

            N_t = len(Nt)
            # Step 2: Initialize charging profiles r_n^0 for active EVs
            for ev_id in Nt:
                ev_data = self.EVs[ev_id]
                d_n = ev_data['deadline']
                horizon_n = d_n - t
                if ev_data['r_nk'] is None:
                    ev_data['r_nk'] = np.zeros(horizon_n)
                else:
                    ev_data['r_nk'] = ev_data['r_nk'][1:]
                    if len(ev_data['r_nk']) < horizon_n:
                        ev_data['r_nk'] = np.append(ev_data['r_nk'], 0.0)

            # Steps 2-4: Perform K iterations
            for k in range(self.K):
                # Utility Control Signal
                total_load = self.D[t:].copy()
                r_sum = np.zeros(len(total_load))
                for ev_id in Nt:
                    ev_data = self.EVs[ev_id]
                    d_n = ev_data['deadline']
                    horizon_n = d_n - t
                    r_nk = ev_data['r_nk']
                    r_sum[:horizon_n] += r_nk
                total_load_k = total_load[:len(r_sum)] + r_sum
                U_prime = total_load_k
                pk = self.gamma / N_t * U_prime

                # EV Charging Profile Update
                for ev_id in Nt:
                    ev_data = self.EVs[ev_id]
                    d_n = ev_data['deadline']
                    horizon_n = d_n - t
                    pk_n = pk[:horizon_n]
                    r_nk = ev_data['r_nk']
                    Rn_t = ev_data['R_t']
                    max_rate = ev_data['max_rate']
                    r_n = cp.Variable(horizon_n)
                    objective = cp.Minimize(cp.sum(cp.multiply(pk_n, r_n)) + 0.5 * cp.sum_squares(r_n - r_nk))
                    constraints = [r_n >= 0, r_n <= max_rate, cp.sum(r_n) == Rn_t]
                    prob = cp.Problem(objective, constraints)
                    prob.solve()
                    if prob.status == 'optimal':
                        ev_data['r_nk'] = r_n.value
                    else:
                        pass  # Keep previous r_nk if not solved

            # Step 4: Set rn(t) ← rn^K(t) for n ∈ Nt
            for ev_id in Nt:
                ev_data = self.EVs[ev_id]
                r_n_t = ev_data['r_nk'][0]
                ev_data['r_n_history'].append(r_n_t)
                ev_data['R_t'] -= r_n_t
                if ev_data['R_t'] <= 1e-6 or ev_data['deadline'] <= t+1:
                    ev_data['active'] = False
                    ev_data['r_nk'] = None

            # Update total load at time t
            total_r_t = sum([self.EVs[ev_id]['r_n_history'][-1] for ev_id in Nt])
            self.total_load[t] = self.D[t] + total_r_t

            # Record convergence history
            variance = np.var(self.total_load[:t+1])
            self.convergence_history.append(variance)

    def plot_results(self):
        """Plot the results of the optimization"""
        N = len(self.ev_id_list)
        r = np.zeros((N, self.T))
        for idx, ev_id in enumerate(self.ev_id_list):
            ev_data = self.EVs[ev_id]
            r_n_history = ev_data['r_n_history']
            r[idx, :len(r_n_history)] = r_n_history

        total_load = self.total_load
        convergence_history = self.convergence_history

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Time labels (15-minute intervals from 20:00 to 06:30)
        time_labels = []
        current_hour = 20
        current_minute = 0
        for _ in range(self.T):
            time_labels.append(f"{current_hour:02d}:{current_minute:02d}")
            current_minute += 15
            if current_minute >= 60:
                current_minute = 0
                current_hour += 1
                if current_hour >= 24:
                    current_hour = 0

        # Plot 1: Individual EV charging profiles
        for idx in range(N):
            ax1.plot(time_labels, r[idx], label=f'EV {idx+1}', alpha=0.5)
        ax1.set_title('Individual EV Charging Profiles')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Charging Rate (kW)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Convergence history
        ax2.plot(convergence_history)
        ax2.set_title('Convergence History (Variance of Total Load)')
        ax2.set_xlabel('Time Slot')
        ax2.set_ylabel('Total Load Variance')
        ax2.grid(True)

        # Plot 3: Total load profile
        ax3.plot(time_labels, self.D, 'k-', label='Base Load', marker='o')
        ax3.plot(time_labels, total_load, 'b--', label='Total Load')
        ax3.set_title('Total Load Profile')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Load (kW)')
        ax3.legend()
        ax3.grid(True)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

def run_realtime_example():
    T = 52  # 30-minute intervals from 20:00 to 06:30

    np.random.seed(0)
    # Base load profile
    D = np.array([
        0.90, 0.90, 0.89, 0.88, 0.85,  # 20:00-21:00
        0.82, 0.78, 0.75, 0.70, 0.65,  # 21:00-22:00
        0.62, 0.58, 0.55, 0.53, 0.52,  # 22:00-23:00
        0.50, 0.48, 0.47, 0.46, 0.45,  # 23:00-00:00
        0.45, 0.44, 0.44, 0.43, 0.43,  # 00:00-01:00
        0.42, 0.42, 0.42, 0.42, 0.42,  # 01:00-02:00
        0.42, 0.42, 0.42, 0.42, 0.42,  # 02:00-03:00
        0.43, 0.43, 0.44, 0.45, 0.47,  # 03:00-04:00
        0.50, 0.52, 0.55, 0.58, 0.61,  # 04:00-05:00
        0.63, 0.65, 0.67, 0.68, 0.67,  # 05:00-06:00
        0.66, 0.65                      # 06:00-06:30
    ]) * 80
    
    beta = 2.0  # Lipschitz constant
    K = 10  # Number of iterations in each time slot

    rt_odc = RealTimeOptimalDecentralizedCharging(T=T, D=D, beta=beta, K=K)

    # Generate random EVs
    N = 21  # Number of EVs
    for n in range(N-1):
        ev_id = n
        arrival = np.random.uniform(0, T // 2) 
        deadline = T  # All EVs must finish charging by the end of the horizon
        energy_requirement = np.random.uniform(5, 25)  # Random energy requirement in kWh
        max_charging_rate = 3.3  # Maximum charging rate in kW
        rt_odc.add_EV(ev_id, arrival, deadline, energy_requirement, max_charging_rate)

    rt_odc.add_EV(20, T // 2, T // 2 + 2, 50, 25)

    rt_odc.run()
    rt_odc.plot_results()

if __name__ == "__main__":
    run_realtime_example()
