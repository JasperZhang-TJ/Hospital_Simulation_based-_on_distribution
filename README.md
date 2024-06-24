# Hospital_Simulation_based-_on_distribution
This is a hospital department utilization and queuing number simulation project based on simpy and different departments with time-varying service time distributions and patient arrival distributions, please check the installation of relevant libraries before running.

Core Logic

1. Simulation Environment Initialization:
   - Create a simulation environment using SimPy, which forms the foundation for simulating the hospital's operational processes.
2. Data Loading:
   - Load necessary configuration data from external files (YAML and Excel), including room configurations, patient information, and arrival frequencies.
3. Entity Creation:
   - Based on the loaded data, create room and patient entities. The rooms support dynamic capacity changes, meaning their capacity can vary over time.
4. Simulation Process Control:
   - Schedule patient arrivals according to the set arrival frequencies and process them based on their treatment sequences.
   - Rooms will process patients according to the set efficiency, recording the usage and queue length at each time point.
5. Results Output:
   - After the simulation, use matplotlib to plot and display the usage and queue length of each room, providing an intuitive performance visualization through charts.

Key Input Files

1. YAML Files (`rooms.yaml` and `patients.yaml`):
   - Room Configuration (`rooms.yaml`): Contains information about each room's name, processing efficiency, etc. (Note: The capacity in this file is not used; it has been changed to be dynamically input through the Room_Capacities table.)
   - Patient Configuration (`patients.yaml`): Contains information about patient types and treatment sequences. The treatment sequence determines the order in which patients need to visit rooms.

2. Excel Files (`Room_Capacities.xlsx` and `Patient_Arrival_Frequencies.xlsx`):
   - Room Capacities (`Room_Capacities.xlsx`): Defines the capacity changes for each room every half-hour throughout the day.
   - Patient Arrival Frequencies (`Patient_Arrival_Frequencies.xlsx`): Specifies the patient arrival frequencies every half-hour throughout the day, used to control the intervals between patient arrivals.


How to Use the Program

1. Prepare and Check Input Files:
   - Ensure that all input files (the two YAML files and the two Excel files) are correctly placed in accessible paths for the program and that the data format meets expectations.

2. Program Configuration:
   - Update the file paths in the program according to the actual paths to ensure the program can correctly read these files.

3. Run the Simulation:
   - Execute the program `hospital-simulation.py`. The program will automatically load data, initialize the simulation environment, start the simulation process, and eventually display the result charts.

4. View and Analyze Results:
   - Analyze the usage and queue length of each room through the generated charts to assess the hospital's operational efficiency.
