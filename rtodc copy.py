import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EVState:
    """Class to track EV state"""
    arrival_time: int
    deadline: int
    required_energy: float
    charging_profile: np.ndarray
    is_active: bool = False
    total_charged: float = 0.0

class RealTimeOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,  # Total time horizon
        D: np.ndarray,  # Base load profile
        r_min: float = 0.0,  # Minimum charging rate
        r_max: float = 3.3,  # Maximum charging rate
        K: int = 100,  # Number of iterations per time slot
        alpha: float = 0.5,  # Step size parameter
        beta: float = 2.0,  # Lipschitz constant
        tolerance: float = 1e-3
    ):
        self.T = T
        self.D = D
        self.r_min = r_min
        self.r_max = r_max
        self.K = K
        self.beta = beta
        self.tolerance = tolerance
        
        # Pick gamma parameter satisfying 0 < gamma < 1/(N*beta)
        self.gamma = alpha / beta  # Will be adjusted based on active EVs
        
        # Dictionary to store EV states
        self.ev_states: Dict[int, EVState] = {}
        self.active_evs: Set[int] = set()
        self.current_time = 0
        
        # Store results for plotting
        self.total_load_history = []
        self.active_ev_counts = []
        
    def utility_prime(self, x: float) -> float:
        """Derivative of utility function U(x) = 0.5 * x^2"""
        return x
    
    def add_ev(
        self,
        ev_id: int,
        arrival_time: int,
        deadline: int,
        required_energy: float
    ):
        """Add a new EV to the system"""
        self.ev_states[ev_id] = EVState(
            arrival_time=arrival_time,
            deadline=deadline,
            required_energy=required_energy,
            charging_profile=np.zeros(self.T)
        )
    
    def update_active_evs(self, t: int):
        """Update the set of active EVs at time slot t"""
        # Remove completed or deadline-reached EVs
        to_remove = set()
        for ev_id in self.active_evs:
            ev = self.ev_states[ev_id]
            if (ev.total_charged >= ev.required_energy or 
                t >= ev.deadline or 
                not ev.is_active):
                to_remove.add(ev_id)
                ev.is_active = False
        
        self.active_evs -= to_remove
        
        # Add newly arrived EVs
        for ev_id, ev in self.ev_states.items():
            if (t >= ev.arrival_time and 
                t < ev.deadline and 
                ev.total_charged < ev.required_energy and 
                not ev.is_active):
                self.active_evs.add(ev_id)
                ev.is_active = True
    
    def calculate_control_signal(self, t: int, r_sum: float) -> float:
        """Calculate control signal p^k using utility function derivative"""
        total_load = self.D[t] + r_sum
        n_active = len(self.active_evs) if len(self.active_evs) > 0 else 1
        self.gamma = 0.5 / (n_active * self.beta)  # Update gamma based on active EVs
        return self.gamma * self.utility_prime(total_load)
    
    def solve_ev_optimization(
        self,
        ev_id: int,
        t: int,
        control_signal: float,
        previous_rate: float
    ) -> float:
        """Solve single EV optimization problem for current time slot"""
        ev = self.ev_states[ev_id]
        
        # Calculate remaining energy needed
        remaining_energy = ev.required_energy - ev.total_charged
        remaining_time = ev.deadline - t
        
        if remaining_time <= 0:
            return 0.0
        
        # Average rate needed to meet requirement
        min_required_rate = remaining_energy / remaining_time if remaining_time > 0 else 0
        
        # Define optimization variable
        r = cp.Variable(1)
        
        # Objective: p^k * r + (1/2)||r - r_prev||^2
        objective = cp.Minimize(
            control_signal * r[0] + 0.5 * cp.sum_squares(r - previous_rate)
        )
        
        # Constraints
        constraints = [
            r >= max(self.r_min, min_required_rate),  # Must meet minimum required rate
            r <= min(self.r_max, remaining_energy)     # Cannot charge more than needed
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                # Safe conversion of scalar value
                return float(r.value.item())
            else:
                return previous_rate
        except:
            return previous_rate
    
    def run_time_slot(self, t: int) -> Dict[int, float]:
        """Execute one time slot of the RTODC algorithm"""
        self.update_active_evs(t)
        
        if not self.active_evs:
            return {}
        
        # Initialize charging rates for this time slot
        current_rates = {
            ev_id: self.ev_states[ev_id].charging_profile[t-1] if t > 0 else 0.0
            for ev_id in self.active_evs
        }
        
        # Iteration loop
        for k in range(self.K):
            # Calculate total charging rate
            r_sum = sum(current_rates.values())
            
            # Calculate control signal
            control_signal = self.calculate_control_signal(t, r_sum)
            
            # Update each active EV's charging rate
            new_rates = {}
            for ev_id in self.active_evs:
                new_rate = self.solve_ev_optimization(
                    ev_id,
                    t,
                    control_signal,
                    current_rates[ev_id]
                )
                new_rates[ev_id] = new_rate
            
            # Check convergence
            rate_changes = [abs(new_rates[ev_id] - current_rates[ev_id]) 
                          for ev_id in self.active_evs]
            if max(rate_changes) < self.tolerance:
                break
            
            # Update rates
            current_rates = new_rates.copy()
        
        # Update EV states with final rates
        for ev_id, rate in current_rates.items():
            self.ev_states[ev_id].charging_profile[t] = rate
            self.ev_states[ev_id].total_charged += rate
        
        return current_rates
    
    def run(self):
        """Run the complete RTODC algorithm"""
        total_load = np.zeros(self.T)
        for t in range(self.T):
            self.current_time = t
            rates = self.run_time_slot(t)
            
            # Store results for plotting
            total_load[t] = self.D[t] + sum(rates.values())
            self.total_load_history.append(total_load[t])
            self.active_ev_counts.append(len(self.active_evs))
            
            # Print progress
            if t % 10 == 0:
                print(f"Time slot {t}: {len(self.active_evs)} active EVs")
    
    def plot_results(self):
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
        
        # Plot 1: Individual EV charging profiles
        for ev_id, ev in self.ev_states.items():
            ax1.plot(time_labels, ev.charging_profile, 
                    label=f'EV {ev_id}', alpha=0.5)
        ax1.set_title('Individual EV Charging Profiles')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Charging Rate (kW)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Total load profile
        ax2.plot(time_labels, self.D, 'k-', label='Base Load', marker='o')
        ax2.plot(time_labels, self.total_load_history, 'b--', 
                label='Total Load')
        ax2.set_title('Load Profiles')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Load (kW)')
        ax2.legend()
        ax2.grid(True)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: Number of active EVs
        ax3.plot(time_labels, self.active_ev_counts)
        ax3.set_title('Number of Active EVs')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Count')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_rtodc_example():
    """Run example simulation with RTODC"""
    T = 52  # 15-minute intervals from 20:00 to 06:30
    np.random.seed(0)
    
    # Base load profile (same as before)
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
    
    # Initialize RTODC
    rtodc = RealTimeOptimalDecentralizedCharging(
        T=T,
        D=D,
        r_max=3.3,
        r_min=0.0,
        K=100,
        alpha=0.5,
        beta=2.0,
        tolerance=1e-6
    )
    
    # Add EVs with random arrival times
    N = 20  # Number of EVs
    for i in range(N):
        # Random arrival time in first half of horizon
        arrival_time = int(np.random.uniform(0, T//2))
        # Deadline in second half of horizon
        # deadline = np.random.randint(max(arrival_time + 10, T//2), T)
        deadline = T
        # Random energy requirement
        required_energy = np.random.uniform(5, 25)
        
        rtodc.add_ev(i, arrival_time, deadline, required_energy)
    
    # Run the algorithm
    rtodc.run()
    rtodc.plot_results()
    
    return rtodc

if __name__ == "__main__":
    rtodc = run_rtodc_example()
# import numpy as np
# import cvxpy as cp
# from typing import List, Tuple, Dict, Set
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# from collections import defaultdict

# @dataclass
# class EVState:
#     """Class to track EV state"""
#     arrival_time: int
#     deadline: int
#     required_energy: float
#     charging_profile: np.ndarray
#     is_active: bool = False
#     total_charged: float = 0.0

# class RealTimeOptimalDecentralizedCharging:
#     def __init__(
#         self,
#         T: int,  # Total time horizon
#         D: np.ndarray,  # Base load profile
#         r_min: float = 0.0,  # Minimum charging rate
#         r_max: float = 3.3,  # Maximum charging rate
#         K: int = 100,  # Number of iterations per time slot
#         alpha: float = 0.1,  # Step size parameter
#         tolerance: float = 1e-3
#     ):
#         self.T = T
#         self.D = D
#         self.r_min = r_min
#         self.r_max = r_max
#         self.K = K
#         self.alpha = alpha
#         self.tolerance = tolerance
        
#         # Dictionary to store EV states
#         self.ev_states: Dict[int, EVState] = {}
#         self.active_evs: Set[int] = set()
#         self.current_time = 0
        
#         # Store results for plotting
#         self.total_load_history = []
#         self.active_ev_counts = []
        
#     def add_ev(
#         self,
#         ev_id: int,
#         arrival_time: int,
#         deadline: int,
#         required_energy: float
#     ):
#         """Add a new EV to the system"""
#         self.ev_states[ev_id] = EVState(
#             arrival_time=arrival_time,
#             deadline=deadline,
#             required_energy=required_energy,
#             charging_profile=np.zeros(self.T)
#         )
    
#     def update_active_evs(self, t: int):
#         """Update the set of active EVs at time slot t"""
#         # Remove completed or deadline-reached EVs
#         to_remove = set()
#         for ev_id in self.active_evs:
#             ev = self.ev_states[ev_id]
#             if (ev.total_charged >= ev.required_energy or 
#                 t >= ev.deadline or 
#                 not ev.is_active):
#                 to_remove.add(ev_id)
#                 ev.is_active = False
        
#         self.active_evs -= to_remove
        
#         # Add newly arrived EVs
#         for ev_id, ev in self.ev_states.items():
#             if (t >= ev.arrival_time and 
#                 t < ev.deadline and 
#                 ev.total_charged < ev.required_energy and 
#                 not ev.is_active):
#                 self.active_evs.add(ev_id)
#                 ev.is_active = True
    
#     def calculate_control_signal(self, t: int, current_rates: Dict[int, float]) -> float:
#         """Calculate control signal lambda(t)"""
#         if not self.active_evs:
#             return 0.0
        
#         total_current_rate = sum(current_rates.values())
#         return (self.D[t] + total_current_rate) / len(self.active_evs)
    
#     def solve_ev_optimization(
#         self,
#         ev_id: int,
#         t: int,
#         control_signal: float,
#         previous_rate: float
#     ) -> float:
#         """Solve single EV optimization problem for current time slot"""
#         ev = self.ev_states[ev_id]
        
#         # Define optimization variable
#         r = cp.Variable(1)
        
#         # Objective: (1/2)(r - r_prev)^2 + lambda*r
#         objective = cp.Minimize(
#             0.5 * cp.square(r - previous_rate) + control_signal * r
#         )
        
#         # Constraints
#         constraints = [
#             r >= self.r_min,
#             r <= self.r_max,
#             r <= ev.required_energy - ev.total_charged  # Cannot charge more than needed
#         ]
        
#         # Solve the problem
#         problem = cp.Problem(objective, constraints)
#         try:
#             problem.solve()
#             if problem.status == 'optimal':
#                 return float(r.value)
#             else:
#                 return previous_rate
#         except:
#             return previous_rate
    
#     def run_time_slot(self, t: int) -> Dict[int, float]:
#         """Execute one time slot of the RTODC algorithm"""
#         self.update_active_evs(t)
        
#         if not self.active_evs:
#             return {}
        
#         # Initialize charging rates for this time slot
#         current_rates = {
#             ev_id: self.ev_states[ev_id].charging_profile[t-1] if t > 0 else 0.0
#             for ev_id in self.active_evs
#         }
        
#         # Iteration loop
#         for k in range(self.K):
#             # Calculate control signal
#             control_signal = self.calculate_control_signal(t, current_rates)
            
#             # Check convergence
#             if abs(control_signal) < self.tolerance:
#                 break
            
#             # Update each active EV's charging rate
#             new_rates = {}
#             for ev_id in self.active_evs:
#                 new_rate = self.solve_ev_optimization(
#                     ev_id,
#                     t,
#                     control_signal,
#                     current_rates[ev_id]
#                 )
#                 new_rates[ev_id] = new_rate
            
#             # Update rates with step size
#             for ev_id in self.active_evs:
#                 current_rates[ev_id] = (
#                     current_rates[ev_id] +
#                     self.alpha * (new_rates[ev_id] - current_rates[ev_id])
#                 )
        
#         # Update EV states with final rates
#         for ev_id, rate in current_rates.items():
#             self.ev_states[ev_id].charging_profile[t] = rate
#             self.ev_states[ev_id].total_charged += rate
        
#         return current_rates
    
#     def run(self):
#         """Run the complete RTODC algorithm"""
#         for t in range(self.T):
#             self.current_time = t
#             rates = self.run_time_slot(t)
            
#             # Store results for plotting
#             total_load = self.D[t] + sum(rates.values())
#             self.total_load_history.append(total_load)
#             self.active_ev_counts.append(len(self.active_evs))
    
#     def plot_results(self):
#         """Plot the results of the optimization"""
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
#         # Time labels
#         time_labels = []
#         current_hour = 20
#         current_minute = 0
#         for _ in range(self.T):
#             time_labels.append(f"{current_hour:02d}:{current_minute:02d}")
#             current_minute += 15
#             if current_minute >= 60:
#                 current_minute = 0
#                 current_hour += 1
#                 if current_hour >= 24:
#                     current_hour = 0
        
#         # Plot 1: Individual EV charging profiles
#         for ev_id, ev in self.ev_states.items():
#             ax1.plot(time_labels, ev.charging_profile, 
#                     label=f'EV {ev_id}', alpha=0.5)
#         ax1.set_title('Individual EV Charging Profiles')
#         ax1.set_xlabel('Time')
#         ax1.set_ylabel('Charging Rate (kW)')
#         ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax1.grid(True)
#         plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
#         # Plot 2: Total load profile
#         ax2.plot(time_labels, self.D, 'k-', label='Base Load', marker='o')
#         ax2.plot(time_labels, self.total_load_history, 'b--', 
#                 label='Total Load')
#         ax2.set_title('Load Profiles')
#         ax2.set_xlabel('Time')
#         ax2.set_ylabel('Load (kW)')
#         ax2.legend()
#         ax2.grid(True)
#         plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
#         # Plot 3: Number of active EVs
#         ax3.plot(time_labels, self.active_ev_counts)
#         ax3.set_title('Number of Active EVs')
#         ax3.set_xlabel('Time')
#         ax3.set_ylabel('Count')
#         ax3.grid(True)
        
#         plt.tight_layout()
#         plt.show()

# def run_rtodc_example():
#     """Run example simulation with RTODC"""
#     T = 52  # 15-minute intervals from 20:00 to 06:30
#     np.random.seed(0)
    
#     # Base load profile (same as before)
#     D = np.array([
#         0.90, 0.90, 0.89, 0.88, 0.85,  # 20:00-21:00
#         0.82, 0.78, 0.75, 0.70, 0.65,  # 21:00-22:00
#         0.62, 0.58, 0.55, 0.53, 0.52,  # 22:00-23:00
#         0.50, 0.48, 0.47, 0.46, 0.45,  # 23:00-00:00
#         0.45, 0.44, 0.44, 0.43, 0.43,  # 00:00-01:00
#         0.42, 0.42, 0.42, 0.42, 0.42,  # 01:00-02:00
#         0.42, 0.42, 0.42, 0.42, 0.42,  # 02:00-03:00
#         0.43, 0.43, 0.44, 0.45, 0.47,  # 03:00-04:00
#         0.50, 0.52, 0.55, 0.58, 0.61,  # 04:00-05:00
#         0.63, 0.65, 0.67, 0.68, 0.67,  # 05:00-06:00
#         0.66, 0.65                      # 06:00-06:30
#     ]) * 80
    
#     # Initialize RTODC
#     rtodc = RealTimeOptimalDecentralizedCharging(
#         T=T,
#         D=D,
#         r_max=3.3,
#         r_min=0.0,
#         K=100,
#         alpha=0.1,
#         tolerance=1e-3
#     )
    
#     # Add EVs with random arrival times
#     N = 20  # Number of EVs
#     for i in range(N):
#         # Random arrival time in first half of horizon
#         arrival_time = np.random.randint(0, T//2)
#         # Deadline in second half of horizon
#         deadline = T
#         # Random energy requirement
#         required_energy = np.random.uniform(5, 25)
        
#         rtodc.add_ev(i, arrival_time, deadline, required_energy)
    
#     # Run the algorithm
#     rtodc.run()
#     rtodc.plot_results()
    
#     return rtodc

# if __name__ == "__main__":
#     rtodc = run_rtodc_example()