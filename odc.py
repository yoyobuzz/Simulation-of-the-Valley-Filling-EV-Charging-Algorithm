import numpy as np
import cvxpy as cp
from typing import List, Tuple
import matplotlib.pyplot as plt

class OptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,  # Total scheduling horizon
        N: int,  # Number of EVs
        D: np.ndarray,  # Base load profile
        beta: float,  # Lipschitz constant
        arrival_times: np.ndarray,  # Arrival time slot for each EV
        departure_times: np.ndarray,  # Departure time slot for each EV
        r_min: float = 0,  # Minimum charging rate
        r_max: float = 3.3,  # Maximum charging rate
        E_target: np.ndarray = None,  # Target energy for each EV
        max_iterations: int = 100,
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
            
        # Pick gamma parameter satisfying 0 < gamma < 1/(N*beta)
        self.gamma = 0.5 / (N * beta)
        
    def utility_prime(self, x: float) -> float:
        """
        Derivative of utility function U(x)
        Using U(x) = 0.5 * x^2
        """
        a = 1
        return a * x
    
    def calculate_control_signal(self, D_t: float, r_sum_t: float) -> float:
        """Calculate control signal p^k(t) using equation (7)"""
        total_load = D_t + r_sum_t
        return self.gamma * self.utility_prime(total_load)
    
    def get_feasible_charging_window(self, n: int) -> List[Tuple[float, float]]:
        """Get feasible charging rates for each time slot based on EV's arrival and departure times"""
        feasible_set = []
        for t in range(self.T):
            if t >= self.arrival_times[n] and t < self.departure_times[n]:
                feasible_set.append((self.r_min, self.r_max))
            else:
                feasible_set.append((0.0, 0.0))  # No charging allowed outside window
        return feasible_set

    def solve_ev_optimization(
        self,
        p_k: np.ndarray,
        r_k: np.ndarray,
        n: int,
        feasible_set_constraints: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Solve the optimization problem for each EV (equation 8)"""
        T = len(p_k)
        r_next = cp.Variable(T)
        
        # Objective function: <p^k, r_n> + (1/2)||r_n - r_n^k||^2
        objective = cp.Minimize(
            p_k @ r_next + 0.5 * cp.sum_squares(r_next - r_k)
        )
        
        # Constraints
        constraints = []
        
        # Charging rate constraints based on feasible windows
        for t in range(T):
            min_rate, max_rate = feasible_set_constraints[t]
            constraints.extend([
                r_next[t] >= min_rate,
                r_next[t] <= max_rate
            ])
        
        # Total energy constraint: must meet target energy requirement
        constraints.append(cp.sum(r_next) == self.E_target[n])
        
        # Solve the problem
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
        """Run the optimal decentralized charging algorithm"""
        # Initialize charging profiles
        r = np.zeros((self.N, self.T))
        
        # Initialize with feasible solutions within charging windows
        for n in range(self.N):
            window_length = self.departure_times[n] - self.arrival_times[n]
            charging_slots = np.zeros(self.T)
            charging_slots[self.arrival_times[n]:self.departure_times[n]] = \
                self.E_target[n] / window_length
            r[n, :] = charging_slots
            
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            r_prev = r.copy()
            
            # Step ii: Calculate control signal
            p_k = np.zeros(self.T)
            for t in range(self.T):
                r_sum_t = np.sum(r[:, t])
                p_k[t] = self.calculate_control_signal(self.D[t], r_sum_t)
            
            # Step iii: Each EV calculates new charging profile
            for n in range(self.N):
                feasible_set = self.get_feasible_charging_window(n)
                r[n] = self.solve_ev_optimization(
                    p_k,
                    r_prev[n],
                    n,
                    feasible_set
                )
            
            # Calculate total load for convergence history
            total_load = self.D + np.sum(r, axis=0)
            convergence_history.append(np.sum(total_load ** 2))
            
            # Check convergence
            if np.max(np.abs(r - r_prev)) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
                
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




def plot_base_load(D: np.ndarray):
    """Plot the base load profile D over the scheduling horizon."""
    T = len(D)
    
    # Time labels (15-minute intervals from 20:00 to 06:30)
    time_labels = []
    current_hour = 20
    current_minute = 0
    for _ in range(T):
        time_labels.append(f"{current_hour:02d}:{current_minute:02d}")
        current_minute += 15
        if current_minute >= 60:
            current_minute = 0
            current_hour += 1
            if current_hour >= 24:
                current_hour = 0
    
    # Plot the base load profile
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, D, 'k-', label='Base Load', marker='o')
    plt.title('Base Load Profile')
    plt.xlabel('Time')
    plt.ylabel('Load (kW)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_example():
    T = 52  # 30-minute intervals from 20:00 to 06:30
    N = 20  # Number of EVs
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
    
    # Generate random arrival and departure times
    arrival_times = np.random.randint(0, T//2, N)  # First half of time horizon
    departure_times = np.random.randint(T//2, T, N)  # Second half of time horizon
    
    # Ensure minimum charging duration
    min_duration = 10  # Minimum 5 hours charging window
    for n in range(N):
        while departure_times[n] - arrival_times[n] < min_duration:
            departure_times[n] = min(departure_times[n] + 4, T)
    
    # Random energy requirements
    E_target = np.random.uniform(5, 25, N)  # kWh
    
    # Initialize and run the algorithm
    odc = OptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=3.3,
        r_min=0.0,
        max_iterations=1000,
        tolerance=1e-3
    )
    
    optimal_profiles = odc.run()
    return optimal_profiles

def run_example1():
    import datetime
    import pandas as pd
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
    
    # Initialize and run the algorithm
    odc = OptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=3.3,
        r_min=0.0,
        max_iterations=1000,
        tolerance=1e-3
    )
    
    optimal_profiles = odc.run()
    return optimal_profiles

if __name__ == "__main__":
    # D = np.array([
    #     0.90, 0.90, 0.89, 0.88, 0.85,  # 20:00-21:00
    #     0.82, 0.78, 0.75, 0.70, 0.65,  # 21:00-22:00
    #     0.62, 0.58, 0.55, 0.53, 0.52,  # 22:00-23:00
    #     0.50, 0.48, 0.47, 0.46, 0.45,  # 23:00-00:00
    #     0.45, 0.44, 0.44, 0.43, 0.43,  # 00:00-01:00
    #     0.42, 0.42, 0.42, 0.42, 0.42,  # 01:00-02:00
    #     0.42, 0.42, 0.42, 0.42, 0.42,  # 02:00-03:00
    #     0.43, 0.43, 0.44, 0.45, 0.47,  # 03:00-04:00
    #     0.50, 0.52, 0.55, 0.58, 0.61,  # 04:00-05:00
    #     0.63, 0.65, 0.67, 0.68, 0.67,  # 05:00-06:00
    #     0.66, 0.65                      # 06:00-06:30
    # ]) * 80
    # plot_base_load(D)
    optimal_profiles = run_example()