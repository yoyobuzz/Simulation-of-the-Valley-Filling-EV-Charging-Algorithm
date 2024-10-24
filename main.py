import numpy as np
import cvxpy as cp
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

class OptimalDecentralizedCharging:
    def __init__(
        self,
        T: int,  # Scheduling horizon
        N: int,  # Number of EVs
        D: np.ndarray,  # Base load profile
        beta: float,  # Lipschitz constant
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
        
        # Energy requirements for each EV (if not provided, set a default)
        if E_target is None:
            self.E_target = np.ones(N) * 10.0 # 30% of max possible charge
        else:
            self.E_target = E_target
            
        # Pick gamma parameter satisfying 0 < gamma < 1/(N*beta)
        self.gamma = 0.5 / (N * beta)
        
    def utility_prime(self, x: float) -> float:
        """
        Derivative of utility function U(x)
        Using U(x) = 0.5 * x^2  where:
        - x is the total load (base + EV charging)
        """
        a = 1
        return a * x
    
    def calculate_control_signal(self, D_t: float, r_sum_t: float) -> float:
        """Calculate control signal p^k(t) using equation (7)"""
        total_load = D_t + r_sum_t
        return self.gamma * self.utility_prime(total_load)
    
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
        
        # Charging rate constraints
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
                return r_k  # Return previous solution if optimization fails
        except:
            return r_k
    
    def run(self) -> np.ndarray:
        """Run the optimal decentralized charging algorithm"""
        # Initialize charging profiles
        r = np.zeros((self.N, self.T))
        
        # Initialize with feasible solutions
        for n in range(self.N):
            # Simple initialization: spread required energy evenly across time slots
            r[n, :] = self.E_target[n] / self.T
            
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
                feasible_set = [(self.r_min, self.r_max) for _ in range(self.T)]
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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Individual EV charging profiles
        time = np.arange(self.T)
        for n in range(self.N):
            ax1.plot(time, r[n], label=f'EV {n+1}')
        ax1.set_title('Individual EV Charging Profiles')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Charging Rate')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Total load profile
        total_load = self.D + np.sum(r, axis=0)
        ax2.plot(time, self.D, label='Base Load', linestyle='--')
        ax2.plot(time, total_load, label='Total Load')
        ax2.set_title('Load Profiles')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Load')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Convergence history
        ax3.plot(convergence_history)
        ax3.set_title('Convergence History')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Total Cost')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_example():
    # Example parameters
    T = 52  
    N = 20   # 3 EVs
    
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
    ]) * 250  # Peak at noon
    beta = 2.0  # Lipschitz constant
    
    # Target energy requirements for each EV (example: different requirements)
    # E_target = np.array([5.0]*N)  # kWh
    E_target = np.random.uniform(5, 20, N) # kWh
    
    # Initialize and run the algorithm
    odc = OptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        E_target=E_target,
        r_max=3.3,
        r_min=0.0
    )
    
    optimal_profiles = odc.run()
    return optimal_profiles

if __name__ == "__main__":
    optimal_profiles = run_example()