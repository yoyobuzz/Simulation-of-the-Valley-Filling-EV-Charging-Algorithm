import numpy as np
import pandas as pd
import datetime
import sys
import matplotlib.pyplot as plt
from algorithms import OptimalDecentralizedCharging, AsynchronousOptimalDecentralizedCharging, RealTimeOptimalDecentralizedCharging


def run_optimal_decentralized_charging_demo() -> np.ndarray:
    """
    Run the Optimal Decentralized Charging (ODC) algorithm with hardcoded example data.

    Returns:
        np.ndarray: Optimal charging profiles for all EVs.
    """
    T = 52
    N = 20
    np.random.seed(0)
    
    D = np.array([
        0.90, 0.90, 0.89, 0.88, 0.85,
        0.82, 0.78, 0.75, 0.70, 0.65,
        0.62, 0.58, 0.55, 0.53, 0.52,
        0.50, 0.48, 0.47, 0.46, 0.45,
        0.45, 0.44, 0.44, 0.43, 0.43,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.43, 0.43, 0.44, 0.45, 0.47,
        0.50, 0.52, 0.55, 0.58, 0.61,
        0.63, 0.65, 0.67, 0.68, 0.67,
        0.66, 0.65
    ]) * 100

    beta = 2.0
    arrival_times = np.random.randint(0, T // 2, N)
    departure_times = np.random.randint(T // 2, T, N)

    min_duration = 10
    for n in range(N):
        while departure_times[n] - arrival_times[n] < min_duration:
            departure_times[n] = min(departure_times[n] + 4, T)

    E_target = np.random.uniform(5, 25, N)

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


def run_optimal_decentralized_charging_from_file(base_load_file: str, ev_info_file: str) -> np.ndarray:
    """
    Run the Optimal Decentralized Charging (ODC) algorithm using data from files.

    Args:
        base_load_file (str): Path to the Excel file containing base load data.
        ev_info_file (str): Path to the Excel file containing EV information.

    Returns:
        np.ndarray: Optimal charging profiles for all EVs.
    """
    try:
        base_load_df = pd.read_excel(base_load_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Base load file '{base_load_file}' not found.")

    required_columns = ['Time', 'Demand_weekends']
    if not all(col in base_load_df.columns for col in required_columns):
        raise ValueError(f"Base load file must contain columns: {required_columns}")

    def extract_start_time(time_str):
        try:
            start_str = time_str.split('-')[0]
            return datetime.datetime.strptime(start_str, '%H:%M').time()
        except Exception as e:
            raise ValueError(f"Invalid time format '{time_str}': {e}")

    base_load_df['Start_Time'] = base_load_df['Time'].apply(extract_start_time)

    start_time = datetime.time(20, 0)
    end_time = datetime.time(9, 0)     
    slots_after_20 = base_load_df[base_load_df['Start_Time'] >= start_time].copy()
    slots_before_09 = base_load_df[base_load_df['Start_Time'] < end_time].copy()

    filtered_df = pd.concat([slots_after_20, slots_before_09], ignore_index=True)

    expected_slots = 52
    actual_slots = len(filtered_df)
    if actual_slots != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} time slots from 20:00 to 09:00, but found {actual_slots}."
        )

    D = filtered_df['Demand_weekends'].values
    T = len(D)

    try:
        ev_info_df = pd.read_excel(ev_info_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"EV information file '{ev_info_file}' not found.")

    ev_required_columns = [
        'EV_ID', 'Arrival_Time', 'Deadline', 
        'Energy_Requirement', 'Max_Charging_Rate'
    ]
    if not all(col in ev_info_df.columns for col in ev_required_columns):
        raise ValueError(f"EV information file must contain columns: {ev_required_columns}")

    arrival_times = ev_info_df['Arrival_Time'].values
    departure_times = ev_info_df['Deadline'].values
    E_target = ev_info_df['Energy_Requirement'].values
    r_max = ev_info_df['Max_Charging_Rate'].values[0]

    if len(arrival_times) != len(departure_times) or len(departure_times) != len(E_target):
        raise ValueError("Inconsistent EV information in the provided file.")

    N = len(arrival_times)

    beta = 2.0
    
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


def run_asynchronous_optimal_decentralized_charging_demo() -> np.ndarray:
    """
    Run the Asynchronous Optimal Decentralized Charging (AODC) algorithm with hardcoded example data.

    Returns:
        np.ndarray: Optimal charging profiles for all EVs.
    """
    T = 52      
    N = 20      
    D = np.array([
        0.90, 0.90, 0.89, 0.88, 0.85,
        0.82, 0.78, 0.75, 0.70, 0.65,
        0.62, 0.58, 0.55, 0.53, 0.52,
        0.50, 0.48, 0.47, 0.46, 0.45,
        0.45, 0.44, 0.44, 0.43, 0.43,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.43, 0.43, 0.44, 0.45, 0.47,
        0.50, 0.52, 0.55, 0.58, 0.61,
        0.63, 0.65, 0.67, 0.68, 0.67,
        0.66, 0.65
    ]) * 100

    beta = 2.0  
    arrival_times = np.random.randint(0, T // 2, N)
    departure_times = np.random.randint(T // 2, T, N)

    min_duration = 20
    for n in range(N):
        while departure_times[n] - arrival_times[n] < min_duration:
            departure_times[n] = min(departure_times[n] + 4, T)

    E_target = np.random.uniform(5, 25, N)
    
    aodc = AsynchronousOptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=3.3,
        r_min=0.0
    )
    
    optimal_profiles = aodc.run()
    return optimal_profiles


def run_asynchronous_optimal_decentralized_charging_from_file(base_load_file: str, ev_info_file: str) -> np.ndarray:
    """
    Run the Asynchronous Optimal Decentralized Charging (AODC) algorithm using data from files.

    Args:
        base_load_file (str): Path to the Excel file containing base load data.
        ev_info_file (str): Path to the Excel file containing EV information.

    Returns:
        np.ndarray: Optimal charging profiles for all EVs.
    """
    try:
        base_load_df = pd.read_excel(base_load_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Base load file '{base_load_file}' not found.")

    required_columns = ['Time', 'Demand_weekends']
    if not all(col in base_load_df.columns for col in required_columns):
        raise ValueError(f"Base load file must contain columns: {required_columns}")

    def extract_start_time(time_str):
        try:
            start_str = time_str.split('-')[0]
            return datetime.datetime.strptime(start_str, '%H:%M').time()
        except Exception as e:
            raise ValueError(f"Invalid time format '{time_str}': {e}")

    base_load_df['Start_Time'] = base_load_df['Time'].apply(extract_start_time)

    start_time = datetime.time(20, 0)
    end_time = datetime.time(9, 0)

    slots_after_20 = base_load_df[base_load_df['Start_Time'] >= start_time].copy()
    slots_before_09 = base_load_df[base_load_df['Start_Time'] < end_time].copy()

    filtered_df = pd.concat([slots_after_20, slots_before_09], ignore_index=True)

    expected_slots = 52
    actual_slots = len(filtered_df)
    if actual_slots != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} time slots from 20:00 to 09:00, but found {actual_slots}."
        )

    D = filtered_df['Demand_weekends'].values
    T = len(D)

    try:
        ev_info_df = pd.read_excel(ev_info_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"EV information file '{ev_info_file}' not found.")

    ev_required_columns = [
        'EV_ID', 'Arrival_Time', 'Deadline',
        'Energy_Requirement', 'Max_Charging_Rate'
    ]
    if not all(col in ev_info_df.columns for col in ev_required_columns):
        raise ValueError(f"EV information file must contain columns: {ev_required_columns}")

    arrival_times = ev_info_df['Arrival_Time'].values
    departure_times = ev_info_df['Deadline'].values
    E_target = ev_info_df['Energy_Requirement'].values
    r_max = ev_info_df['Max_Charging_Rate'].values[0]

    if len(arrival_times) != len(departure_times) or len(departure_times) != len(E_target):
        raise ValueError("Inconsistent EV information in the provided file.")

    N = len(arrival_times)

    beta = 2.0
    
    aodc = AsynchronousOptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=r_max,
        r_min=0.0
    )

    optimal_profiles = aodc.run()
    return optimal_profiles


def run_realtime_optimal_decentralized_charging_demo() -> None:
    """
    Run the Real-Time Optimal Decentralized Charging (RTODC) algorithm with hardcoded example data.
    """
    T = 52
    np.random.seed(0)

    D = np.array([
        0.90, 0.90, 0.89, 0.88, 0.85,
        0.82, 0.78, 0.75, 0.70, 0.65,
        0.62, 0.58, 0.55, 0.53, 0.52,
        0.50, 0.48, 0.47, 0.46, 0.45,
        0.45, 0.44, 0.44, 0.43, 0.43,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.43, 0.43, 0.44, 0.45, 0.47,
        0.50, 0.52, 0.55, 0.58, 0.61,
        0.63, 0.65, 0.67, 0.68, 0.67,
        0.66, 0.65
    ]) * 100

    beta = 2.0
    K = 10  
    rt_odc = RealTimeOptimalDecentralizedCharging(T=T, D=D, beta=beta, K=K)

    N = 21
    for n in range(N - 1):
        ev_id = n
        arrival = np.random.uniform(0, T // 2)
        deadline = T
        energy_requirement = np.random.uniform(5, 25)
        max_charging_rate = 3.3
        rt_odc.add_EV(ev_id, arrival, deadline, energy_requirement, max_charging_rate)

    rt_odc.add_EV(20, T // 2, T // 2 + 2, 50, 25)

    rt_odc.run()
    rt_odc.plot_results()


def run_realtime_optimal_decentralized_charging_from_file(base_load_file: str, ev_info_file: str) -> None:
    """
    Run the Real-Time Optimal Decentralized Charging (RTODC) algorithm using data from files.

    Args:
        base_load_file (str): Path to the Excel file containing base load data.
        ev_info_file (str): Path to the Excel file containing EV information.
    """
    try:
        base_load_df = pd.read_excel(base_load_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Base load file '{base_load_file}' not found.")

    required_columns = ['Time', 'Demand_weekends']
    if not all(col in base_load_df.columns for col in required_columns):
        raise ValueError(f"Base load file must contain columns: {required_columns}")

    def extract_start_time(time_str):
        try:
            start_str = time_str.split('-')[0]
            return datetime.datetime.strptime(start_str, '%H:%M').time()
        except Exception as e:
            raise ValueError(f"Invalid time format '{time_str}': {e}")

    base_load_df['Start_Time'] = base_load_df['Time'].apply(extract_start_time)

    start_time = datetime.time(20, 0)      
    end_time = datetime.time(9, 0)     
    slots_after_20 = base_load_df[base_load_df['Start_Time'] >= start_time].copy()
    slots_before_09 = base_load_df[base_load_df['Start_Time'] < end_time].copy()

    filtered_df = pd.concat([slots_after_20, slots_before_09], ignore_index=True)

    expected_slots = 52
    actual_slots = len(filtered_df)
    if actual_slots != expected_slots:
        raise ValueError(
            f"Expected {expected_slots} time slots from 20:00 to 09:00, but found {actual_slots}."
        )

    D = filtered_df['Demand_weekends'].values
    T = len(D)

    beta = 2.0      
    K = 10     
    rt_odc = RealTimeOptimalDecentralizedCharging(T=T, D=D, beta=beta, K=K)

    try:
        ev_info_df = pd.read_excel(ev_info_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"EV information file '{ev_info_file}' not found.")

    ev_required_columns = [
        'EV_ID', 'Arrival_Time', 'Deadline', 
        'Energy_Requirement', 'Max_Charging_Rate'
    ]
    if not all(col in ev_info_df.columns for col in ev_required_columns):
        raise ValueError(f"EV information file must contain columns: {ev_required_columns}")

    for index, row in ev_info_df.iterrows():
        try:
            ev_id = int(row['EV_ID'])
            arrival = int(row['Arrival_Time'])
            deadline = int(row['Deadline'])
            energy_requirement = float(row['Energy_Requirement'])
            max_charging_rate = float(row['Max_Charging_Rate'])
            rt_odc.add_EV(ev_id, arrival, deadline, energy_requirement, max_charging_rate)
        except ValueError as e:
            print(
                f"Error processing EV information at row {index + 2}: {e}", 
                file=sys.stderr
            )

    rt_odc.run()
    rt_odc.plot_results()


def plot_convergence_comparison(odc_convergence: list, aodc_convergence: list) -> None:
    """
    Plot the convergence comparison between ODC and AODC algorithms.

    Args:
        odc_convergence (list): Convergence history for ODC algorithm.
        aodc_convergence (list): Convergence history for AODC algorithm.
    """
    plt.figure(figsize=(10, 6))
    iterations_odc = list(range(1, len(odc_convergence) + 1))
    iterations_aodc = list(range(1, len(aodc_convergence) + 1))

    plt.plot(iterations_odc, odc_convergence, 'b--', label='Algorithm ODC', linewidth=2)
    plt.plot(iterations_aodc, aodc_convergence, 'r-', label='Algorithm AODC', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of iterations', fontsize=14)
    plt.ylabel(r'$||R^k - R^*||/N$ (kW)', fontsize=14)
    plt.title('Convergence Comparison of ODC and AODC Algorithms', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=300)
    plt.show()



def run_and_plot_convergence() -> None:
    """
    Run the Optimal Decentralized Charging (ODC) algorithm with hardcoded example data.

    Returns:
        np.ndarray: Optimal charging profiles for all EVs.
    """
    T = 52
    N = 20
    np.random.seed(0)

    D = np.array([
        0.90, 0.90, 0.89, 0.88, 0.85,
        0.82, 0.78, 0.75, 0.70, 0.65,
        0.62, 0.58, 0.55, 0.53, 0.52,
        0.50, 0.48, 0.47, 0.46, 0.45,
        0.45, 0.44, 0.44, 0.43, 0.43,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.42,
        0.43, 0.43, 0.44, 0.45, 0.47,
        0.50, 0.52, 0.55, 0.58, 0.61,
        0.63, 0.65, 0.67, 0.68, 0.67,
        0.66, 0.65
    ]) * 80

    beta = 2.0
    arrival_times = np.random.randint(0, T // 2, N)
    departure_times = np.random.randint(T // 2, T, N)

    min_duration = 10
    for n in range(N):
        while departure_times[n] - arrival_times[n] < min_duration:
            departure_times[n] = min(departure_times[n] + 4, T)

    E_target = np.random.uniform(5, 25, N)

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
    odc_convergence = odc.convergence_history
    odc_convergence = odc.convergence_history

    # Run AODC algorithm and get convergence history
    aodc = AsynchronousOptimalDecentralizedCharging(
        T=T,
        N=N,
        D=D,
        beta=beta,
        arrival_times=arrival_times,
        departure_times=departure_times,
        E_target=E_target,
        r_max=3.3,
        r_min=0.0
    )
    optimal_profiles = aodc.run()
    aodc_convergence = aodc.convergence_history
    aodc_convergence = aodc.convergence_history

    # Plot convergence comparison
    plot_convergence_comparison(odc_convergence, aodc_convergence)


if __name__ == "__main__":
    # Uncomment the function you want to run:

    # Run Optimal Decentralized Charging (ODC) with demo data
    # optimal_profiles_odc_demo = run_optimal_decentralized_charging_demo()
    # print("Optimal charging profiles (ODC demo):", optimal_profiles_odc_demo)

    # Run Optimal Decentralized Charging (ODC) with data from files
    # optimal_profiles_odc_file = run_optimal_decentralized_charging_from_file("base_load.xlsx", "ev_info.xlsx")
    # print("Optimal charging profiles (ODC file):", optimal_profiles_odc_file)

    # Run Asynchronous Optimal Decentralized Charging (AODC) with demo data
    # optimal_profiles_aodc_demo = run_asynchronous_optimal_decentralized_charging_demo()
    # print("Optimal charging profiles (AODC demo):", optimal_profiles_aodc_demo)

    # Run Asynchronous Optimal Decentralized Charging (AODC) with data from files
    # optimal_profiles_aodc_file = run_asynchronous_optimal_decentralized_charging_from_file("base_load.xlsx", "ev_info.xlsx")
    # print("Optimal charging profiles (AODC file):", optimal_profiles_aodc_file)

    # Run Real-Time Optimal Decentralized Charging (RTODC) with demo data
    # run_realtime_optimal_decentralized_charging_demo()

    # Run Real-Time Optimal Decentralized Charging (RTODC) with data from files
    # run_realtime_optimal_decentralized_charging_from_file("base_load.xlsx", "ev_info.xlsx")

    # Run convergence comparison between ODC and AODC
    run_and_plot_convergence()