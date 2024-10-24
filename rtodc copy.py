import numpy as np
import cvxpy as cp
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

@dataclass
class EVState:
    """Class to track EV state"""
    is_active: bool
    deadline: int
    remaining_energy: float
    charging_profile: np.ndarray
    r_max: float
    previous_rate: float  # Store previous iteration's rate

class RealTimeOptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,
        D: np.ndarray,
        gamma: float,
        K: int,  # Number of iterations per time slot
        r_max: float = 3.3,
        epsilon: float = 1e-3
    ):
        self.T = T
        self.D = D
        self.gamma = gamma
        self.K = K
        self.r_max = r_max
        self.epsilon = epsilon
        
        # State tracking
        self.current_time = 0
        self.active_evs: Dict[int, EVState] = {}
        self.charging_history: List[Dict[int, float]] = []
        
    def activate_ev(
        self,
        ev_id: int,
        deadline: int,
        energy_required: float
    ):
        """Activate a new EV for charging"""
        if deadline <= self.current_time:
            print(f"Warning: EV {ev_id} deadline {deadline} not after current time {self.current_time}")
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
        
    def calculate_control_signal(self, current_rates: Dict[int, float]) -> float:
        """Calculate control signal λ(t) for current time slot"""
        if not self.active_evs:
            return 0.0
            
        total_charging = sum(current_rates.values())
        return (self.D[self.current_time] - total_charging) / len(self.active_evs)
    
    def optimize_ev_rate(
        self,
        ev_id: int,
        lambda_t: float,
        previous_rate: float
    ) -> float:
        """Optimize charging rate for single EV at current time slot"""
        ev = self.active_evs[ev_id]
        
        # Define variable for current time slot
        r = cp.Variable(1)
        
        # Following Algorithm 3's optimization objective
        objective = cp.Minimize(
            0.5 * cp.square(r[0] - previous_rate) + lambda_t * r[0]
        )
        
        # Calculate minimum required rate to meet deadline
        remaining_time = ev.deadline - self.current_time
        if remaining_time > 0:
            min_rate = max(0, ev.remaining_energy / remaining_time)
        else:
            min_rate = 0
        
        constraints = [
            r >= min_rate,
            r <= min(ev.r_max, ev.remaining_energy)
        ]
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status == 'optimal':
                return float(r.value)
            return previous_rate
        except Exception as e:
            print(f"Optimization failed for EV {ev_id}: {e}")
            return previous_rate
    
    def step_time(self):
        """Process one time slot"""
        if self.current_time >= self.T:
            return False
        
        # Initialize current rates dictionary
        current_rates = {ev_id: 0.0 for ev_id in self.active_evs}
        
        # Store previous rates
        for ev_id, ev in self.active_evs.items():
            if self.charging_history and ev_id in self.charging_history[-1]:
                ev.previous_rate = self.charging_history[-1][ev_id]
            else:
                ev.previous_rate = 0.0
            current_rates[ev_id] = ev.previous_rate
        
        # Execute K iterations of optimization
        for k in range(self.K):
            # Calculate control signal
            lambda_t = self.calculate_control_signal(current_rates)
            # print(f"Time {self.current_time}, Iteration {k}, Control signal: {lambda_t:.4f}")
            
            if abs(lambda_t) < self.epsilon:
                # Small control signal - apply direct update
                for ev_id in self.active_evs:
                    current_rate = current_rates[ev_id]
                    new_rate = current_rate + self.gamma * lambda_t
                    # Ensure rate stays within bounds
                    ev = self.active_evs[ev_id]
                    new_rate = max(0, min(new_rate, ev.r_max, ev.remaining_energy))
                    current_rates[ev_id] = new_rate
            else:
                # Optimize rates
                old_rates = current_rates.copy()
                for ev_id, ev in self.active_evs.items():
                    new_rate = self.optimize_ev_rate(
                        ev_id,
                        lambda_t,
                        old_rates[ev_id]
                    )
                    current_rates[ev_id] = new_rate
            
            # # Print some rates for debugging
            # print(f"Sample rates after iteration {k}: {dict(list(current_rates.items())[:3])}")
        
        # Update charging profiles and states
        charging_decisions = {}
        for ev_id, rate in current_rates.items():
            ev = self.active_evs[ev_id]
            ev.charging_profile[self.current_time] = rate
            charging_decisions[ev_id] = rate
            
            # Update remaining energy
            ev.remaining_energy -= rate
            
            # Check if EV should become inactive
            if (ev.remaining_energy <= 0 or 
                self.current_time >= ev.deadline):
                ev.is_active = False
        
        # Clean up inactive EVs
        inactive_evs = [
            ev_id for ev_id, ev in self.active_evs.items() 
            if not ev.is_active
        ]
        for ev_id in inactive_evs:
            self.deactivate_ev(ev_id)
        
        self.charging_history.append(charging_decisions)
        self.current_time += 1
        return True
    
    def deactivate_ev(self, ev_id: int):
        """Deactivate an EV"""
        if ev_id in self.active_evs:
            del self.active_evs[ev_id]

    def run_simulation(
        self,
        arrival_schedule: List[Tuple[int, int, float, int]]
    ):
        """
        Run full simulation with given arrival schedule
        arrival_schedule: List of (time, ev_id, energy_required, deadline)
        """
        arrival_schedule.sort(key=lambda x: x[0])
        
        while self.current_time < self.T:
            # Check for new arrivals
            for arrival in arrival_schedule:
                arrival_time, ev_id, energy, deadline = arrival
                if arrival_time == self.current_time:
                    self.activate_ev(ev_id, deadline, energy)
            
            # Process time step
            self.step_time()
            
            if self.current_time % 10 == 0:
                print(f"\nProcessed time {self.current_time}")
                print(f"Active EVs: {len(self.active_evs)}")
                if self.charging_history:
                    print(f"Total charging rate: {sum(self.charging_history[-1].values()):.2f}")
        
        return self.charging_history
    
    def plot_results(self):
        """Plot simulation results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Calculate total load at each time step
        total_loads = []
        ev_charging_loads = []
        for t in range(self.T):
            if t < len(self.charging_history):
                ev_load = sum(self.charging_history[t].values())
            else:
                ev_load = 0
            total_loads.append(self.D[t] + ev_load)
            ev_charging_loads.append(ev_load)
        
        # Plot loads
        time_range = range(self.T)
        ax1.plot(time_range, self.D, 'k-', label='Base Load')
        ax1.plot(time_range, total_loads, 'r--', label='Total Load')
        ax1.plot(time_range, ev_charging_loads, 'b:', label='EV Charging Load')
        ax1.set_title('Load Profiles')
        ax1.set_xlabel('Time Slot')
        ax1.set_ylabel('Load (kW)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot individual EV charging profiles
        ev_profiles = {}
        for t, decisions in enumerate(self.charging_history):
            for ev_id, rate in decisions.items():
                if ev_id not in ev_profiles:
                    ev_profiles[ev_id] = np.zeros(self.T)
                ev_profiles[ev_id][t] = rate
        
        for ev_id, profile in ev_profiles.items():
            ax2.plot(time_range, profile, alpha=0.5, label=f'EV {ev_id}')
        
        ax2.set_title('Individual EV Charging Profiles')
        ax2.set_xlabel('Time Slot')
        ax2.set_ylabel('Charging Rate (kW)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
def run_example():
    """Run example simulation of decentralized EV charging"""
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    # Parameters
    T = 52  # 15-minute intervals from 20:00 to 06:30
    
    # Base load profile (scaled)
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
    
    # Initialize algorithm with proper parameters
    rtoc = RealTimeOptimalDecentralizedCharging(
        T=T,
        D=D,
        gamma=0.1,  # Changed from gamma to gamma
        K=10,
        r_max=3.3  # Maximum charging rate (kW)
    )
    
    # Generate more realistic arrival schedule
    N = 20  # Number of EVs
    arrival_schedule = []
    
    # Different EV types with their typical battery capacities
    ev_types = [
        {"capacity": 40, "charging_rate": 3.3},  # Small EV
        {"capacity": 60, "charging_rate": 3.3},  # Medium EV
        {"capacity": 85, "charging_rate": 3.3}   # Large EV
    ]
    
    for i in range(N):
        # Random arrival time in evening hours (20:00-23:00)
        arrival_time = random.randint(0, 12)  # First 3 hours
        
        # Select random EV type
        ev_type = random.choice(ev_types)
        
        # Calculate realistic energy requirement based on EV type
        # Assume EVs arrive with 20-40% remaining charge
        remaining_charge = random.uniform(0.2, 0.4)
        energy_needed = ev_type["capacity"] * (0.8 - remaining_charge)  # Charge up to 80%
        
        # Set deadline based on typical overnight charging scenario
        min_charging_slots = int(np.ceil(energy_needed / ev_type["charging_rate"]))
        min_deadline = min(arrival_time + min_charging_slots, T-1)
        deadline = random.randint(min_deadline, T)  # Must finish by 6:30 AM
        
        arrival_schedule.append((arrival_time, i, energy_needed, deadline))
    
    # Sort schedule by arrival time
    arrival_schedule.sort(key=lambda x: x[0])
    
    # Run simulation
    while rtoc.current_time < T:
        # Process new arrivals for current time step
        current_arrivals = [
            (t, ev_id, energy, deadline) 
            for t, ev_id, energy, deadline in arrival_schedule 
            if t == rtoc.current_time
        ]
        
        # Activate new EVs
        for _, ev_id, energy, deadline in current_arrivals:
            rtoc.activate_ev(ev_id, deadline, energy)
        
        # Process time step
        rtoc.step_time()
        
        if rtoc.current_time % 10 == 0:
            print(f"\nTime: {rtoc.current_time} (Hour {20 + rtoc.current_time//4}:{(rtoc.current_time%4)*15:02d})")
            print(f"Active EVs: {len(rtoc.active_evs)}")
            if rtoc.charging_history:
                total_rate = sum(rtoc.charging_history[-1].values())
                print(f"Total charging rate: {total_rate:.2f} kW")
                print(f"Base load: {D[rtoc.current_time]:.2f} kW")
                print(f"Total load: {(D[rtoc.current_time] + total_rate):.2f} kW")
    
    rtoc.plot_results()
    
    return rtoc.charging_history

if __name__ == "__main__":
    charging_history = run_example()