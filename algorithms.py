import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random

class OptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        N: int,
        D: np.ndarray,
        beta: float,
        arrival_times: np.ndarray,
        departure_times: np.ndarray,
        r_min: float = 0,
        r_max: float = 3.3,
        E_target: np.ndarray = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.T = T
        self.N = N
        self.D = D
        self.beta = beta
        self.r_min = r_min
        self.r_max = r_max
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.arrival_times = arrival_times
        self.departure_times = departure_times

        if E_target is None:
            self.E_target = np.ones(N) * 10.0
        else:
            self.E_target = E_target

        self.gamma = 0.5 / (N * beta)

    def utility_prime(self, x: float) -> float:
        return x

    def calculate_control_signal(self, D_t: float, r_sum_t: float) -> float:
        total_load = D_t + r_sum_t
        return self.gamma * self.utility_prime(total_load)

    def get_feasible_charging_window(self, n: int) -> List[Tuple[float, float]]:
        feasible_set = []
        for t in range(self.T):
            if self.arrival_times[n] <= t < self.departure_times[n]:
                feasible_set.append((self.r_min, self.r_max))
            else:
                feasible_set.append((0.0, 0.0))
        return feasible_set

    def solve_ev_optimization(
        self,
        p_k: np.ndarray,
        r_k: np.ndarray,
        n: int,
        feasible_set_constraints: List[Tuple[float, float]],
    ) -> np.ndarray:
        T = len(p_k)
        r_next = cp.Variable(T)

        objective = cp.Minimize(p_k @ r_next + 0.5 * cp.sum_squares(r_next - r_k))

        constraints = []
        for t in range(T):
            min_rate, max_rate = feasible_set_constraints[t]
            constraints.extend([r_next[t] >= min_rate, r_next[t] <= max_rate])
        constraints.append(cp.sum(r_next) == self.E_target[n])

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                return r_next.value
            else:
                return r_k
        except Exception as e:
            print(f"Optimization error for EV {n}: {e}")
            return r_k

    def run(self) -> np.ndarray:
        r = np.zeros((self.N, self.T))

        for n in range(self.N):
            window_length = self.departure_times[n] - self.arrival_times[n]
            if window_length > 0:
                charging_slots = np.zeros(self.T)
                charging_slots[self.arrival_times[n] : self.departure_times[n]] = (
                    self.E_target[n] / window_length
                )
                r[n, :] = charging_slots

        convergence_history = []

        for iteration in range(self.max_iterations):
            r_prev = r.copy()

            p_k = np.zeros(self.T)
            for t in range(self.T):
                r_sum_t = np.sum(r[:, t])
                p_k[t] = self.calculate_control_signal(self.D[t], r_sum_t)

            for n in range(self.N):
                feasible_set = self.get_feasible_charging_window(n)
                r[n] = self.solve_ev_optimization(p_k, r_prev[n], n, feasible_set)

            total_load = self.D + np.sum(r, axis=0)
            convergence_history.append(np.sum(total_load ** 2))

            if np.max(np.abs(r - r_prev)) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        self.plot_results(r, convergence_history)
        return r

    def plot_results(self, r: np.ndarray, convergence_history: List[float]):
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
                    current_hour -= 24

        x_ticks = list(range(0, self.T, 16))
        x_tick_labels = [time_labels[i] for i in x_ticks]

        plt.figure(figsize=(7, 5))
        for n in range(self.N):
            plt.plot(r[n], label=f'EV {n + 1}', alpha=0.6, linewidth=1.2)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Individual EV Charging Profiles', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Charging Rate (kW)', fontsize=14)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("individual_ev_profiles.png", dpi=300)
        plt.show()

        plt.figure(figsize=(7, 5))
        total_load = self.D + np.sum(r, axis=0)
        plt.plot(self.D, 'k-', label='Base Load', marker='o', linewidth=1.5, markersize=5)
        plt.plot(total_load, 'b--', label='Total Load', marker='x', linewidth=1.5, markersize=5)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Load Profiles', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Total Load (kW/household)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("load_profiles.png", dpi=300)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(convergence_history, 'r-', linewidth=2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Convergence History', fontsize=16, fontweight='bold')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Cost', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("convergence_history.png", dpi=300)
        plt.show()

class AsynchronousOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        N: int,
        D: np.ndarray,
        beta: float,
        arrival_times: np.ndarray,
        departure_times: np.ndarray,
        d: int = 3,
        r_min: float = 0,
        r_max: float = 3.3,
        E_target: np.ndarray = None,
        max_iterations: int = 400,
        tolerance: float = 1e-3,
    ):
        self.T = T
        self.N = N
        self.D = D
        self.beta = beta
        self.d = d
        self.r_min = r_min
        self.r_max = r_max
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.arrival_times = arrival_times
        self.departure_times = departure_times

        if E_target is None:
            self.E_target = np.ones(N) * 10.0
        else:
            self.E_target = E_target

        self.gamma = 0.8 / (N * beta * (3 * d + 1))

        self.K_0 = self._initialize_utility_update_set()
        self.K_n = self._initialize_ev_update_sets()

        self.r_history = []
        self.p_history = []

    def _initialize_utility_update_set(self) -> List[int]:
        return list(range(0, self.max_iterations, self.d))

    def _initialize_ev_update_sets(self) -> Dict[int, List[int]]:
        K_n = {}
        for n in range(self.N):
            offset = random.randint(0, self.d - 1)
            K_n[n] = list(range(offset, self.max_iterations, self.d))
        return K_n

    def utility_prime(self, x: float) -> float:
        return self.beta * x

    def calculate_control_signal(self, k: int) -> np.ndarray:
        p_k = np.zeros(self.T)

        if k == 0 or (k - 1) in self.K_0:
            for t in range(self.T):
                total_load = self.D[t]
                if self.r_history:
                    for n in range(self.N):
                        delay = random.randint(0, self.d)
                        r_delayed = self.get_delayed_charging_profile(self.r_history[-1], k, delay)
                        total_load += r_delayed[n, t]
                p_k[t] = self.gamma * self.utility_prime(total_load)

        return p_k

    def get_delayed_charging_profile(self, r: np.ndarray, k: int, delay: int) -> np.ndarray:
        delayed_k = max(0, k - delay)
        if delayed_k == 0 or delayed_k >= len(self.r_history):
            return np.zeros_like(r)
        return self.r_history[delayed_k]

    def solve_ev_optimization(self, p_k: np.ndarray, r_k: np.ndarray, n: int) -> np.ndarray:
        r_next = cp.Variable(self.T)
        constraints = []
        for t in range(self.T):
            if self.arrival_times[n] <= t < self.departure_times[n]:
                constraints.extend([r_next[t] >= self.r_min, r_next[t] <= self.r_max])
            else:
                constraints.append(r_next[t] == 0)
        constraints.append(cp.sum(r_next) == self.E_target[n])

        objective = cp.Minimize(p_k @ r_next + 0.5 * cp.sum_squares(r_next - r_k))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status == 'optimal':
            return r_next.value
        return r_k

    def run(self) -> np.ndarray:
        r = np.zeros((self.N, self.T))
        self.r_history = [r.copy()]
        self.p_history = [np.zeros(self.T)]

        convergence_history = []
        no_significant_change_count = 0

        for k in range(self.max_iterations):
            r_prev = r.copy()

            p_k = self.calculate_control_signal(k)
            self.p_history.append(p_k)

            for n in range(self.N):
                if k in self.K_n[n]:
                    delay = random.randint(0, self.d)
                    p_delayed = self.calculate_control_signal(max(0, k - delay))
                    r[n] = self.solve_ev_optimization(p_delayed, r_prev[n], n)

            self.r_history.append(r.copy())

            total_load = self.D + np.sum(r, axis=0)
            convergence_history.append(np.sum(total_load ** 2))

            max_change = np.max(np.abs(r - r_prev))
            if max_change < self.tolerance:
                no_significant_change_count += 1
            else:
                no_significant_change_count = 0

            if no_significant_change_count >= 2 * self.d:
                break

        self.plot_results(r, convergence_history)
        return r

    def plot_results(self, r: np.ndarray, convergence_history: List[float]):
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
                    current_hour -= 24

        x_ticks = list(range(0, self.T, 16))
        x_tick_labels = [time_labels[i] for i in x_ticks]

        plt.figure(figsize=(7, 5))
        for n in range(self.N):
            plt.plot(r[n], label=f'EV {n + 1}', alpha=0.6, linewidth=1.2)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Individual EV Charging Profiles', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Charging Rate (kW)', fontsize=14)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        total_load = self.D + np.sum(r, axis=0)
        plt.plot(self.D, 'k-', label='Base Load', marker='o', linewidth=1.5, markersize=5)
        plt.plot(total_load, 'b--', label='Total Load', marker='x', linewidth=1.5, markersize=5)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Load Profiles', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Total Load (kW/household)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(convergence_history, 'r-', linewidth=2)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Convergence History', fontsize=16, fontweight='bold')
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Cost', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

class RealTimeOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        D: np.ndarray,
        beta: float,
        K: int,
    ):
        self.T = T
        self.D = D
        self.beta = beta
        self.K = K
        self.gamma = 0.5 / beta
        self.t = 0
        self.EVs: Dict[int, Dict] = {}
        self.Nt: List[int] = []
        self.total_load = np.zeros(self.T)
        self.convergence_history = []
        self.ev_id_list = []

    def add_EV(
        self,
        ev_id: int,
        arrival_time: int,
        deadline: int,
        energy_needed: float,
        max_charging_rate: float,
    ):
        ev_data = {
            'id': ev_id,
            'arrival': arrival_time,
            'deadline': deadline,
            'R_t': energy_needed,
            'max_rate': max_charging_rate,
            'active': False,
            'r_nk': None,
            'r_n_history': [],
        }
        self.EVs[ev_id] = ev_data
        self.ev_id_list.append(ev_id)

    def run(self):
        for t in range(self.T):
            self.t = t

            Nt = [ev_id for ev_id, ev_data in self.EVs.items() if ev_data['arrival'] <= t < ev_data['deadline'] and ev_data['R_t'] > 1e-6]
            self.Nt = Nt

            if not Nt:
                self.total_load[t] = self.D[t]
                self.convergence_history.append(np.var(self.total_load[:t + 1]))
                continue

            N_t = len(Nt)

            for ev_id in Nt:
                ev_data = self.EVs[ev_id]
                d_n = ev_data['deadline']
                horizon_n = d_n - t
                if ev_data['r_nk'] is None:
                    ev_data['r_nk'] = np.zeros(horizon_n)
                else:
                    ev_data['r_nk'] = ev_data['r_nk'][1:]
                    ev_data['r_nk'] = np.append(ev_data['r_nk'], 0.0)

            for k in range(self.K):
                total_load = self.D[t:].copy()
                r_sum = np.zeros(len(total_load))
                for ev_id in Nt:
                    ev_data = self.EVs[ev_id]
                    d_n = ev_data['deadline']
                    horizon_n = d_n - t
                    r_sum[:horizon_n] += ev_data['r_nk']
                total_load_k = total_load[:len(r_sum)] + r_sum
                U_prime = total_load_k
                pk = self.gamma / N_t * U_prime

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
                    if prob.status == cp.OPTIMAL:
                        ev_data['r_nk'] = r_n.value

            for ev_id in Nt:
                ev_data = self.EVs[ev_id]
                r_n_t = ev_data['r_nk'][0]
                ev_data['r_n_history'].append(r_n_t)
                ev_data['R_t'] = max(0, ev_data['R_t'] - r_n_t)
                if ev_data['R_t'] <= 1e-6 or ev_data['deadline'] <= t + 1:
                    ev_data['active'] = False
                    ev_data['r_nk'] = None

            total_r_t = sum([self.EVs[ev_id]['r_n_history'][-1] for ev_id in Nt])
            self.total_load[t] = self.D[t] + total_r_t

            variance = np.var(self.total_load[:t + 1])
            self.convergence_history.append(variance)

    def plot_results(self):
        N = len(self.ev_id_list)
        r = np.zeros((N, self.T))

        for idx, ev_id in enumerate(self.ev_id_list):
            ev_data = self.EVs[ev_id]
            r_n_history = ev_data['r_n_history']
            arrival = ev_data['arrival']
            arrival_int = int(arrival)
            charging_duration = len(r_n_history)
            charging_end = min(arrival_int + charging_duration, self.T)
            r[idx, arrival_int:charging_end] = r_n_history[:charging_end - arrival_int]

        total_load = self.total_load
        convergence_history = self.convergence_history

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
                    current_hour -= 24

        x_ticks = list(range(0, self.T, 16))
        x_tick_labels = [time_labels[i] for i in x_ticks]

        plt.figure(figsize=(7, 5))
        for idx in range(N):
            plt.plot(r[idx], label=f'EV {idx + 1}', alpha=0.6, linewidth=1.2)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Individual EV Charging Profiles', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Charging Rate (kW)', fontsize=14)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("individual_ev_profiles_rtodc.png", dpi=300)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(range(1, self.T + 1), convergence_history, marker='o', linewidth=1.5)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Convergence History (Variance of Total Load)', fontsize=16, fontweight='bold')
        plt.xlabel('Time Slot', fontsize=14)
        plt.ylabel('Total Load Variance', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("convergence_history_rtodc.png", dpi=300)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.plot(self.D, 'k-', label='Base Load', marker='o', linewidth=1.5, markersize=5)
        plt.plot(total_load, 'b--', label='Total Load', marker='x', linewidth=1.5, markersize=5)
        plt.xticks(x_ticks, x_tick_labels, rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Total Load Profile', fontsize=16, fontweight='bold')
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Load (kW)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("total_load_profile_rtodc.png", dpi=300)
        plt.show()