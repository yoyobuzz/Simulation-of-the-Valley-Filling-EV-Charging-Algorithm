import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class EVState:
    """Class to track EV state"""
    is_active: bool
    deadline: int
    remaining_energy: float
    charging_profile: np.ndarray
    r_max: float
    previous_rate: float

class RealTimeOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        D: np.ndarray,
        gamma: float,
        K: int,
        r_max: float = 3.3,
    ):
        self.T = T
        self.D = D
        self.gamma = gamma
        self.K = K
        self.r_max = r_max
        self.current_time = 0
        self.active_evs: Dict[int, EVState] = {}
        self.charging_history: List[Dict[int, float]] = []

    def utility_derivative(self, total_load: float) -> float:
        """
        Compute derivative of utility function.
        Using quadratic utility U(x) = -0.5(x - target)^2 for valley filling
        """
        # Target load level for valley filling (can be adjusted)
        target_load = 0.76  # This matches the figure's optimal level
        return total_load - target_load

    def calculate_control_signal(self, current_rates: Dict[int, float], time_window: int) -> float:
        """Calculate control signal p^k for valley filling"""
        if not self.active_evs:
            return 0.0
        
        total_load = self.D[time_window] + sum(current_rates.values())
        Nt = len(self.active_evs)
        
        # Control signal based on utility derivative
        return (self.gamma / Nt) * self.utility_derivative(total_load)

    def optimize_ev_rate(
        self,
        ev_id: int,
        control_signals: np.ndarray,
        previous_rates: np.ndarray
    ) -> np.ndarray:
        """Optimize charging profile for single EV for valley filling"""
        ev = self.active_evs[ev_id]
        remaining_time = ev.deadline - self.current_time
        
        # Define optimization variables
        r = cp.Variable(remaining_time)
        
        # Valley filling objective
        objective_terms = []
        for t in range(remaining_time):
            # Control signal term from utility derivative
            objective_terms.append(control_signals[t] * r[t])
            # Deviation penalty
            objective_terms.append(0.5 * cp.square(r[t] - previous_rates[t]))
        
        objective = cp.Minimize(sum(objective_terms))
        
        # Constraints
        constraints = [
            r >= 0,  # Non-negative charging rate
            r <= ev.r_max,  # Maximum charging rate
            cp.sum(r) == ev.remaining_energy  # Must meet energy requirement
        ]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                return r.value
            return previous_rates
        except Exception as e:
            print(f"Optimization failed for EV {ev_id}: {e}")
            return previous_rates

    def step_time(self):
        """Process one time slot with valley filling objective"""
        if self.current_time >= self.T:
            return False
            
        # Initialize rates from previous time slot
        current_rates = {
            ev_id: (
                self.charging_history[-1][ev_id] if self.charging_history and ev_id in self.charging_history[-1]
                else 0.0
            )
            for ev_id in self.active_evs
        }
        
        # Iterative optimization
        for k in range(self.K):
            # Calculate control signals for remaining time window
            control_signals = np.zeros(self.T - self.current_time)
            for t in range(len(control_signals)):
                control_signals[t] = self.calculate_control_signal(
                    current_rates,
                    self.current_time + t
                )
            
            # Update each EV's charging profile
            for ev_id, ev in self.active_evs.items():
                previous_rates = np.zeros(ev.deadline - self.current_time)
                previous_rates[0] = current_rates[ev_id]
                
                new_rates = self.optimize_ev_rate(
                    ev_id,
                    control_signals[:ev.deadline - self.current_time],
                    previous_rates
                )
                current_rates[ev_id] = new_rates[0]
        
        # Update states and check completion
        charging_decisions = {}
        inactive_evs = []
        
        for ev_id, rate in current_rates.items():
            ev = self.active_evs[ev_id]
            charging_decisions[ev_id] = rate
            ev.remaining_energy -= rate
            
            if ev.remaining_energy <= 0 or self.current_time >= ev.deadline:
                inactive_evs.append(ev_id)
        
        for ev_id in inactive_evs:
            del self.active_evs[ev_id]
            
        self.charging_history.append(charging_decisions)
        self.current_time += 1
        return True

    def activate_ev(self, ev_id: int, deadline: int, energy_required: float):
        """Activate new EV with valley-filling consideration"""
        if deadline <= self.current_time:
            return False
            
        profile = np.zeros(self.T)
        self.active_evs[ev_id] = EVState(
            is_active=True,
            deadline=deadline,
            remaining_energy=energy_required,
            charging_profile=profile,
            r_max=self.r_max,
            previous_rate=0.0
        )
        return True
    
def run_example():
    """Run example simulation with valley-filling behavior"""
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    # Parameters matching the paper's figure
    T = 48  # 15-minute intervals from 20:00 to 8:00
    
    # Base load profile (normalized as in the paper)
    D = np.array([
        # 20:00-00:00 (evening peak and decline)
        0.90, 0.90, 0.89, 0.88, 0.87, 0.85, 0.82, 0.78, 0.75, 0.70, 0.65, 0.62,
        # 00:00-04:00 (valley period)
        0.58, 0.55, 0.52, 0.50, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.42,
        # 04:00-08:00 (morning rise)
        0.42, 0.43, 0.44, 0.45, 0.47, 0.50, 0.52, 0.55, 0.58, 0.61, 0.63, 0.65,
        0.67, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60, 0.59, 0.58
    ])
    
    # Initialize algorithm
    rtoc = RealTimeOptimalDecentralizedCharging(
        T=T,
        D=D,
        gamma=0.1,
        K=20,  # Increased iterations for better convergence
        r_max=3.3
    )
    
    # Generate EVs arriving uniformly between 20:00 and 23:00
    N = 30  # Number of EVs
    arrival_schedule = []
    
    # Uniform arrivals in the evening (20:00-23:00)
    arrival_times = np.linspace(0, 20, N, dtype=int)  # First 3 hours
    
    for i, arrival_time in enumerate(arrival_times):
        # Each EV needs about 10 kWh (adjusted to achieve the target level)
        energy = random.uniform(8, 12)
        deadline = T  # All EVs must finish by 8:00
        arrival_schedule.append((arrival_time, i, energy, deadline))
    
    # Run simulation
    while rtoc.current_time < T:
        # Process new arrivals
        current_arrivals = [
            (t, ev_id, energy, deadline) 
            for t, ev_id, energy, deadline in arrival_schedule 
            if t == rtoc.current_time
        ]
        
        for _, ev_id, energy, deadline in current_arrivals:
            rtoc.activate_ev(ev_id, deadline, energy)
        
        rtoc.step_time()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Time labels
    time_range = range(T)
    time_labels = [f"{20+t//4:02d}:00" if t%4==0 else "" for t in range(T)]
    
    # Plot base load
    plt.plot(time_range, D, 'k-', label='Base load', linewidth=2)
    
    # Plot total load
    total_loads = []
    for t in range(T):
        if t < len(rtoc.charging_history):
            ev_load = sum(rtoc.charging_history[t].values())
        else:
            ev_load = 0
        total_loads.append(D[t] + ev_load)
    
    plt.plot(time_range, total_loads, 'r--', label='Algorithm RTODC', linewidth=2)
    
    # Plot settings
    plt.xlabel('time of day')
    plt.ylabel('total load (kW/household)')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, T, 4), [label for i, label in enumerate(time_labels) if i % 4 == 0])
    plt.ylim(0.4, 1.0)
    
    plt.title('EVs plug in uniformly between 20:00 and 23:00')
    plt.show()
    
    return rtoc.charging_history

if __name__ == "__main__":
    charging_history = run_example()