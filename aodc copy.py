import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class AsynchronousOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,  # Total scheduling horizon
        N: int,  # Number of EVs
        D: np.ndarray,  # Base load profile
        beta: float,  # Parameter for utility function
        arrival_times: np.ndarray,  # Arrival time slot for each EV
        departure_times: np.ndarray,  # Departure time slot for each EV
        r_min: float = 0,  # Minimum charging rate
        r_max: float = 3.3,  # Maximum charging rate
        E_target: np.ndarray = None,  # Target energy for each EV
        max_iterations: int = 10000,
        tolerance: float = 1e-6
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
        
        # Energy requirements for each EV
        if E_target is None:
            self.E_target = np.ones(N) * 10.0
        else:
            self.E_target = E_target
            
        # Pick gamma parameter satisfying 0 < gamma < 1/(N*beta*(3d + 1))
        # where d is maximum delay (we'll use d=1 for simplicity)
        d = 1
        self.gamma = 0.5 / (N * beta * (3*d + 1))
        
        # Track last update time for each EV
        self.last_update = np.zeros(N)
        
        # Initialize K_n sets (time slots when each EV can update)
        self.K_n = self._initialize_update_sets()
        
    def _initialize_update_sets(self) -> Dict[int, List[int]]:
        """Initialize the set of time slots when each EV can update"""
        K_n = defaultdict(list)
        for n in range(self.N):
            # Each EV can update in time slots between arrival and departure
            update_slots = list(range(int(self.arrival_times[n]), int(self.departure_times[n])))
            K_n[n] = update_slots
        return K_n
    
    def utility_prime(self, x: float) -> float:
        """
        Derivative of utility function U(x)
        Using U(x) = (beta/2) * x^2 as in the paper
        """
        return self.beta * x
    
    def calculate_control_signal(self, D_t: float, r_sum_t: float) -> float:
        """Calculate control signal p^k(t) using equation from Algorithm 2"""
        total_load = D_t + r_sum_t
        return self.gamma * self.utility_prime(total_load)
    
    def get_feasible_charging_window(self, n: int) -> List[Tuple[float, float]]:
        """Get feasible charging rates for each time slot based on EV's arrival and departure times"""
        feasible_set = []
        for t in range(self.T):
            if t >= self.arrival_times[n] and t < self.departure_times[n]:
                feasible_set.append((self.r_min, self.r_max))
            else:
                feasible_set.append((0.0, 0.0))
        return feasible_set

    def solve_ev_optimization(
        self,
        p_k: np.ndarray,
        r_k: np.ndarray,
        n: int,
        feasible_set_constraints: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Solve the optimization problem for each EV (equation from Algorithm 2)"""
        T = len(p_k)
        r_next = cp.Variable(T)
        
        # Objective function from Algorithm 2
        objective = cp.Minimize(
            p_k @ r_next + 0.5 * cp.sum_squares(r_next - r_k)
        )
        
        constraints = []
        
        # Charging rate constraints
        for t in range(T):
            min_rate, max_rate = feasible_set_constraints[t]
            constraints.extend([
                r_next[t] >= min_rate,
                r_next[t] <= max_rate
            ])
        
        # Total energy constraint
        constraints.append(cp.sum(r_next) == self.E_target[n])
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                return r_next.value
            else:
                return r_k
        except:
            return r_k
    
    def run(self) -> np.ndarray:
        """Run the asynchronous optimal decentralized charging algorithm"""
        # Initialize charging profiles
        r = np.zeros((self.N, self.T))
        
        # Initialize with feasible solutions
        for n in range(self.N):
            window_length = self.departure_times[n] - self.arrival_times[n]
            charging_slots = np.zeros(self.T)
            charging_slots[self.arrival_times[n]:self.departure_times[n]] = \
                self.E_target[n] / window_length
            r[n, :] = charging_slots
            
        convergence_history = []
        
        # Track changes for each EV over multiple iterations
        ev_changes = {n: [] for n in range(self.N)}
        convergence_window = 5  # Number of iterations to check for stability
        
        for iteration in range(self.max_iterations):
            r_prev = r.copy()
            
            # Calculate control signal
            p_k = np.zeros(self.T)
            for t in range(self.T):
                r_sum_t = np.sum(r[:, t])
                p_k[t] = self.calculate_control_signal(self.D[t], r_sum_t)
            
            # Randomly select EVs to update (asynchronous updates)
            active_evs = random.sample(range(self.N), max(1, self.N // 3))
            
            # Track which EVs updated this iteration
            updated_this_iteration = set()
            
            for n in active_evs:
                # Check if EV can update at current iteration
                if iteration in self.K_n[n]:
                    feasible_set = self.get_feasible_charging_window(n)
                    r[n] = self.solve_ev_optimization(
                        p_k,
                        r_prev[n],
                        n,
                        feasible_set
                    )
                    self.last_update[n] = iteration
                    updated_this_iteration.add(n)
                    
                    # Track change for this EV
                    change = np.max(np.abs(r[n] - r_prev[n]))
                    ev_changes[n].append(change)
                    # Keep only last convergence_window changes
                    ev_changes[n] = ev_changes[n][-convergence_window:]
            
            # Calculate total load for convergence history
            total_load = self.D + np.sum(r, axis=0)
            convergence_history.append(np.sum(total_load ** 2))
            
            # Check convergence: all EVs must be stable over their recent updates
            all_evs_stable = True
            for n in range(self.N):
                # Skip EVs that haven't had enough updates yet
                if len(ev_changes[n]) < convergence_window:
                    all_evs_stable = False
                    break
                    
                # Check if EV's recent changes are all below tolerance
                if max(ev_changes[n]) >= self.tolerance:
                    all_evs_stable = False
                    break
                    
                # Check if EV has updated recently
                iterations_since_update = iteration - self.last_update[n]
                if iterations_since_update > 2 * self.N:  # Ensure recent update
                    all_evs_stable = False
                    break
            
            if all_evs_stable:
                print(f"Converged after {iteration + 1} iterations")
                print(f"Final changes per EV: {[max(changes) if changes else float('inf') for changes in ev_changes.values()]}")
                break
                
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, max change: {max([max(changes) if changes else float('inf') for changes in ev_changes.values()])}")
        
        self.plot_results(r, convergence_history)
        return r
    # def run(self) -> np.ndarray:
    #     """Run the asynchronous optimal decentralized charging algorithm"""
    #     # Initialize charging profiles
    #     r = np.zeros((self.N, self.T))
        
    #     # Initialize with feasible solutions
    #     for n in range(self.N):
    #         window_length = self.departure_times[n] - self.arrival_times[n]
    #         charging_slots = np.zeros(self.T)
    #         charging_slots[self.arrival_times[n]:self.departure_times[n]] = \
    #             self.E_target[n] / window_length
    #         r[n, :] = charging_slots
            
    #     convergence_history = []
        
    #     for iteration in range(self.max_iterations):
    #         r_prev = r.copy()
            
    #         # Calculate control signal
    #         p_k = np.zeros(self.T)
    #         for t in range(self.T):
    #             r_sum_t = np.sum(r[:, t])
    #             p_k[t] = self.calculate_control_signal(self.D[t], r_sum_t)
            
    #         # Randomly select EVs to update (asynchronous updates)
    #         active_evs = random.sample(range(self.N), max(1, self.N // 3))
            
    #         for n in active_evs:
    #             # Check if EV can update at current iteration
    #             if iteration in self.K_n[n]:
    #                 feasible_set = self.get_feasible_charging_window(n)
    #                 r[n] = self.solve_ev_optimization(
    #                     p_k,
    #                     r_prev[n],
    #                     n,
    #                     feasible_set
    #                 )
    #                 self.last_update[n] = iteration
            
    #         # Calculate total load for convergence history
    #         total_load = self.D + np.sum(r, axis=0)
    #         convergence_history.append(np.sum(total_load ** 2))
            
    #         # Check convergence
    #         if np.max(np.abs(r - r_prev)) < self.tolerance and iteration in self.K_n[n]:
    #             print(f"Converged after {iteration + 1} iterations")
    #             break
                
    #     self.plot_results(r, convergence_history)
    #     return r
    
    def plot_results(self, r: np.ndarray, convergence_history: List[float]):
        """Plot the results of the optimization"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Time labels
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
        
        # Plot individual EV charging profiles
        for n in range(self.N):
            ax1.plot(time_labels, r[n], label=f'EV {n+1}', alpha=0.5)
        ax1.set_title('Individual EV Charging Profiles')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Charging Rate (kW)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot load profiles
        total_load = self.D + np.sum(r, axis=0)
        ax2.plot(time_labels, self.D, 'k-', label='Base Load', marker='o')
        ax2.plot(time_labels, total_load, 'b--', label='Total Load')
        ax2.set_title('Load Profiles')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Load (kW)')
        ax2.legend()
        ax2.grid(True)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot convergence history
        ax3.plot(convergence_history)
        ax3.set_title('Convergence History')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Total Cost')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_example():
    # Parameters from the paper
    T = 52  # 15-minute intervals from 20:00 to 06:30
    N = 10  # Number of EVs
    
    # Base load profile (scaled as in paper)
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
    ]) * 80  # Scaled to match paper's base load
    
    beta = 2.0  # From paper's utility function
    
    # Generate random arrival and departure times
    # arrival_times = np.random.randint(0, T//2, N)
    # departure_times = np.random.randint(T//2, T, N)
    arrival_times = E_target = np.array([0]*N)
    departure_times = E_target = np.array([T]*N)
    
    # Ensure minimum charging duration (5 hours)
    min_duration = 20  # 20 15-minute intervals = 5 hours
    for n in range(N):
        while departure_times[n] - arrival_times[n] < min_duration:
            departure_times[n] = min(departure_times[n] + 4, T)
    
    # Energy requirements (between 5 and 25 kWh as in paper)
    E_target = np.random.uniform(5, 25, N)
    
    # Initialize and run algorithm
    aodc = AsynchronousOptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=3.3,  # Maximum charging rate from paper
        r_min=0.0
    )
    
    optimal_profiles = aodc.run()
    return optimal_profiles

if __name__ == "__main__":
    optimal_profiles = run_example()