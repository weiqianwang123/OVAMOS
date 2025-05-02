import numpy as np
import matplotlib.pyplot as plt
import csv

# Read from CSV file
csv_filename = "/home/yfx/vlfm/output_frames_POMDP/path.csv"

robot_path = []
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        robot_path.append([float(row[0]), float(row[1])])  # Read (r_px, c_px), ignoring theta

robot_path = np.array(robot_path)  # Convert to NumPy array
r_px, c_px = robot_path[:, 0], robot_path[:, 1]

# Assign Z values: Points further in the path have a higher Z value
z_values = np.arange(len(r_px))  # Increment Z for each new point

# Plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the robot path
ax.scatter(r_px, c_px, z_values, c='r', marker='o', label='Robot Path')

# Connect the points with a line
ax.plot(r_px, c_px, z_values, 'b--', label="Path Line")

# Labels and visualization settings
ax.set_xlabel('r_px')
ax.set_ylabel('c_px')
ax.set_zlabel('Z (Path Progression)')
ax.set_title('Robot Path in 3D')

ax.legend()
plt.show()
