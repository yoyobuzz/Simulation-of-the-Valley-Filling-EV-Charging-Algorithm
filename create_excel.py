
import pandas as pd
import numpy as np
import datetime
def export_hardcoded_data_to_excel(base_load_file='base_load.xlsx', ev_info_file='ev_info.xlsx'):
    """
    Exports hardcoded base load and EV information data to Excel files.

    Parameters:
    - base_load_file (str): The filename for the base load Excel file.
    - ev_info_file (str): The filename for the EV information Excel file.
    """
    # ------------------------------
    # Export Base Load Data
    # ------------------------------
    
    # Define the number of time slots and ensure it matches 15-minute intervals from 20:00 to 09:00
    T = 52  # 15-minute intervals over 13 hours (20:00 to 09:00)
    
    # Generate time slots from 20:00 to 09:00 with 15-minute intervals
    start_datetime = datetime.datetime.strptime("20:00", "%H:%M")
    time_slots = [
        (start_datetime + datetime.timedelta(minutes=15*i)).strftime("%H:%M") + "-" +
        (start_datetime + datetime.timedelta(minutes=15*(i+1))).strftime("%H:%M")
        for i in range(T)
    ]
    
    # Hardcoded Demand_weekends data scaled by 80
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
    ]) * 80  # Scaling factor as per original code
    
    # Validate the length of D
    if len(D) != T:
        raise ValueError(f"Length of Demand_weekends (D) must be {T}, but got {len(D)}.")
    
    # Create DataFrame for base load
    base_load_df = pd.DataFrame({
        'Time': time_slots,
        'Demand_weekends': D
    })
    
    # Export to Excel
    base_load_df.to_excel(base_load_file, index=False, engine='openpyxl')
    print(f"Base load data exported successfully to '{base_load_file}'.")
    
    # ------------------------------
    # Export EV Information Data
    # ------------------------------
    
    # Define parameters
    beta = 2.0  # Lipschitz constant
    K = 10      # Number of iterations in each time slot
    N = 21      # Number of EVs
    np.random.seed(0)  # For reproducibility
    
    # Initialize lists to store EV information
    ev_ids = []
    arrival_times = []
    deadlines = []
    energy_requirements = []
    max_charging_rates = []
    
    # Generate random EVs
    for n in range(N-1):
        ev_id = n
        arrival = int(np.random.uniform(0, T // 2))  # Assuming T//2 is the midpoint
        deadline = T  # All EVs must finish charging by the end of the horizon
        energy_requirement = round(float(np.random.uniform(5, 25)), 2)  # Rounded for clarity
        max_charging_rate = 3.3  # Maximum charging rate in kW
        
        ev_ids.append(ev_id)
        arrival_times.append(arrival)
        deadlines.append(deadline)
        energy_requirements.append(energy_requirement)
        max_charging_rates.append(max_charging_rate)
    
    # Add the last EV with specific parameters
    ev_ids.append(20)
    arrival_times.append(int(T // 2))
    deadlines.append(int(T // 2 + 2))
    energy_requirements.append(50.0)
    max_charging_rates.append(25.0)
    
    # Create DataFrame for EV information
    ev_info_df = pd.DataFrame({
        'EV_ID': ev_ids,
        'Arrival_Time': arrival_times,
        'Deadline': deadlines,
        'Energy_Requirement': energy_requirements,
        'Max_Charging_Rate': max_charging_rates
    })
    
    # Export to Excel
    ev_info_df.to_excel(ev_info_file, index=False, engine='openpyxl')
    print(f"EV information data exported successfully to '{ev_info_file}'.")
export_hardcoded_data_to_excel()