import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import random
import datetime
import pandas as pd

class AsynchronousOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        N: int,
        D: np.ndarray,
        beta: float,
        arrival_times: np.ndarray,
        departure_times: np.ndarray,
        d: int = 3,  # Maximum delay bound
        r_min: float = 0,
        r_max: float = 3.3,
        E_target: np.ndarray = None,
        max_iterations: int = 400,
        tolerance: float = 1e-3
    ):
        self.T = T
        self.N = N
        self.D = D
        self.beta = beta
        self.d = d  # Maximum delay bound
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
            
        # Pick gamma parameter satisfying 0 < gamma < 1/(N*beta*(3d + 1))
        self.gamma = 0.8 / (N * beta * (3*d + 1))
        
        # Initialize K_0 (utility update iterations) and K_n (EV update iterations)
        self.K_0 = self._initialize_utility_update_set()
        self.K_n = self._initialize_ev_update_sets()
        
        # Store charging profiles history for delay simulation
        self.r_history = []
        self.p_history = []
        
    def _initialize_utility_update_set(self) -> List[int]:
        """Initialize set of iterations when utility updates control signal"""
        # Utility updates every d iterations
        return list(range(0, self.max_iterations, self.d))
    
    def _initialize_ev_update_sets(self) -> Dict[int, List[int]]:
        """Initialize sets of iterations when EVs update their profiles"""
        K_n = {}
        for n in range(self.N):
            # Each EV updates every d iterations with random offset
            offset = random.randint(0, self.d-1)
            K_n[n] = list(range(offset, self.max_iterations, self.d))
        return K_n
    
    def utility_prime(self, x: float) -> float:
        """Derivative of utility function U(x) = (beta/2) * x^2"""
        return self.beta * x
    
    def get_delayed_charging_profile(self, r: np.ndarray, k: int, delay: int) -> np.ndarray:
        """Get delayed charging profile r^(k-b) based on history"""
        delayed_k = max(0, k - delay)
        if delayed_k == 0:
            return np.zeros_like(r)
        return self.r_history[delayed_k]
    
    def get_delayed_control_signal(self, k: int, delay: int) -> np.ndarray:
        """Get delayed control signal p^(k-a) based on history"""
        delayed_k = max(0, k - delay)
        if delayed_k == 0 or not self.p_history:
            return np.zeros(self.T)
        return self.p_history[delayed_k]
    
    def calculate_control_signal(self, k: int) -> np.ndarray:
        """Calculate control signal p^k considering delays"""
        p_k = np.zeros(self.T)
        
        if k == 0 or (k-1) in self.K_0:
            for t in range(self.T):
                # Get delayed charging profiles for each EV
                total_load = self.D[t]
                for n in range(self.N):
                    delay = random.randint(0, self.d)  # b_n(k)
                    r_delayed = self.get_delayed_charging_profile(self.r_history[-1], k, delay)
                    total_load += r_delayed[n, t]
                
                p_k[t] = self.gamma * self.utility_prime(total_load)
        
        return p_k
    
    def solve_ev_optimization(
        self,
        p_k: np.ndarray,
        r_k: np.ndarray,
        n: int
    ) -> np.ndarray:
        """Solve the optimization problem for each EV with delayed control signal"""
        r_next = cp.Variable(self.T)
        
        # Get feasible charging window
        constraints = []
        for t in range(self.T):
            if t >= self.arrival_times[n] and t < self.departure_times[n]:
                constraints.extend([
                    r_next[t] >= self.r_min,
                    r_next[t] <= self.r_max
                ])
            else:
                constraints.extend([
                    r_next[t] == 0
                ])
        
        # Total energy constraint
        constraints.append(cp.sum(r_next) == self.E_target[n])
        
        # Objective function from Algorithm 2
        objective = cp.Minimize(
            p_k @ r_next + 0.5 * cp.sum_squares(r_next - r_k)
        )
        
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
        # Initialize charging profiles to zero as per Algorithm 2
        r = np.zeros((self.N, self.T))
        self.r_history = [r.copy()]
        self.p_history = [np.zeros(self.T)]
        
        convergence_history = []
        no_significant_change_count = 0
        
        for k in range(self.max_iterations):
            r_prev = r.copy()
            
            # Step ii: Calculate control signal with delays
            p_k = self.calculate_control_signal(k)
            self.p_history.append(p_k)
            
            # Step iii: Update charging profiles for eligible EVs
            for n in range(self.N):
                if k in self.K_n[n]:
                    # Get delayed control signal
                    delay = random.randint(0, self.d)  # a_n(k)
                    p_delayed = self.get_delayed_control_signal(k, delay)
                    
                    # Update charging profile
                    r[n] = self.solve_ev_optimization(
                        p_delayed,
                        r_prev[n],
                        n
                    )
            
            self.r_history.append(r.copy())
            
            # Calculate total load for convergence history
            total_load = self.D + np.sum(r, axis=0)
            convergence_history.append(np.sum(total_load ** 2))
            
            # Check convergence
            max_change = np.max(np.abs(r - r_prev))
            if max_change < self.tolerance:
                no_significant_change_count += 1
            else:
                no_significant_change_count = 0
                
            # Require stable solution for several consecutive iterations
            if no_significant_change_count >= 2 * self.d:
                print(f"Converged after {k + 1} iterations")
                break
            
            if k % 100 == 0:
                print(f"Iteration {k}, max change: {max_change}")
        
        self.plot_results(r, convergence_history)
        return r
    
    def plot_results(self, r: np.ndarray, convergence_history: List[float]):
        """Plot the results of the optimization"""

        # Time labels (4-hour intervals from 20:00 to 08:00)
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

        # Use every 16th label (4-hour intervals) for x-axis ticks
        x_ticks = list(range(0, self.T, 16))
        x_tick_labels = [time_labels[i] for i in x_ticks]

        # Plot 1: Individual EV Charging Profiles
        plt.figure(figsize=(7, 5))
        for n in range(self.N):
            plt.plot(r[n], label=f'EV {n+1}', alpha=0.6, linewidth=1.2)
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

        # Plot 2: Total Load Profile
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

        # Plot 3: Convergence History
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

def run_example1():
    import datetime
    # Read base load data from Excel
    base_load_file = 'base_load.xlsx'  # Replace with your actual file path
    try:
        base_load_df = pd.read_excel(base_load_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Base load file '{base_load_file}' not found.")

    # Ensure required columns exist
    required_columns = ['Time', 'Demand_weekends']
    if not all(col in base_load_df.columns for col in required_columns):
        raise ValueError(f"Base load file must contain columns: {required_columns}")

    # Parse the 'Time' column to extract start times as datetime.time objects
    def extract_start_time(time_str):
        try:
            start_str = time_str.split('-')[0]
            return datetime.datetime.strptime(start_str, '%H:%M').time()
        except Exception as e:
            raise ValueError(f"Invalid time format '{time_str}': {e}")

    base_load_df['Start_Time'] = base_load_df['Time'].apply(extract_start_time)

    # Define start and end times for filtering
    start_time = datetime.time(20, 0)  # 20:00
    end_time = datetime.time(9, 0)     # 09:00

    # Filter rows where Start_Time >= 20:00 or Start_Time < 09:00
    slots_after_20 = base_load_df[base_load_df['Start_Time'] >= start_time].copy()
    slots_before_09 = base_load_df[base_load_df['Start_Time'] < end_time].copy()

    # Concatenate to ensure 20:00 slots come first
    filtered_df = pd.concat([slots_after_20, slots_before_09], ignore_index=True)

    # Verify that exactly 52 slots are selected
    expected_slots = 52
    actual_slots = len(filtered_df)
    if actual_slots != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} time slots from 20:00 to 09:00, but found {actual_slots}."
        )

    # Extract D and time_labels from the filtered DataFrame
    D = filtered_df['Demand_weekends'].values
    T = len(D)  # Total scheduling horizon based on the number of time slots

    # Read EV information from Excel
    ev_info_file = 'ev_info.xlsx'  # Replace with your actual file path
    try:
        ev_info_df = pd.read_excel(ev_info_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"EV information file '{ev_info_file}' not found.")

    # Ensure required columns exist in EV info
    ev_required_columns = [
        'EV_ID', 'Arrival_Time', 'Deadline', 
        'Energy_Requirement', 'Max_Charging_Rate'
    ]
    
    if not all(col in ev_info_df.columns for col in ev_required_columns):
        raise ValueError(f"EV information file must contain columns: {ev_required_columns}")

    # Extract EV data
    arrival_times = ev_info_df['Arrival_Time'].values
    departure_times = ev_info_df['Deadline'].values
    E_target = ev_info_df['Energy_Requirement'].values
    r_max = ev_info_df['Max_Charging_Rate'].values[0]  # Assuming uniform max charging rate

    # Validate data consistency
    if len(arrival_times) != len(departure_times) or len(departure_times) != len(E_target):
        raise ValueError("Inconsistent EV information in the provided file.")

    N = len(arrival_times)  # Number of EVs

    # Initialize and run algorithm
    beta = 2.0  # Lipschitz constant from paper
    aodc = AsynchronousOptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=r_max,  # Maximum charging rate from EV data
        r_min=0.0
    )

    optimal_profiles = aodc.run()
    return optimal_profiles

def run_example():
    # Parameters from the paper
    T = 52  # 15-minute intervals from 20:00 to 06:30
    N = 20  # Number of EVs
    
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
    arrival_times = np.random.randint(0, T//2, N)
    departure_times = np.random.randint(T//2, T, N)
    # arrival_times = E_target = np.array([0]*N)
    # departure_times = E_target = np.array([T]*N)
    
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

# if __name__ == "__main__":
#     optimal_profiles = run_example1()