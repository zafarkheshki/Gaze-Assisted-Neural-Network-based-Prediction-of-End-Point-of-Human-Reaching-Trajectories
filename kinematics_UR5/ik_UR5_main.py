

from numpy import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import forward_kinem_UR5
import optim_ik_UR5
from transforms3d import *
import time
import csv



my_data = genfromtxt('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.csv', delimiter = ',')

# XYZ positions of end effector
x_end = my_data[:,0] 
y_end = my_data[:,1] 
z_end = my_data[:,2] 

# Quaternions
x_quat_end = my_data[:,3]
y_quat_end = my_data[:,4]
z_quat_end = my_data[:,5]
w_quat_end = my_data[:,6]

# For each joint q_max and q_min values
theta_min = array([-180, -180, -180, -180, -180, -180, -180])*pi/180
theta_max = array([ 180, 180, 180, 180, 180, 180, 180])*pi/180 

theta_actual = random.uniform(theta_min, theta_max, 7)
theta_init = (theta_min + theta_max)

x_e_compute_array = []
y_e_compute_array = []
z_e_compute_array = []

q = []

for i in range(len(my_data)):

	# Positions 
	x_e_actual = x_end[i]
	y_e_actual = y_end[i]
	z_e_actual = z_end[i]

	# Quaternions
	qx = x_quat_end[i]
	qy = y_quat_end[i]
	qz = z_quat_end[i]
	qw = w_quat_end[i] 

	
	R_e_actual = array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
	[2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw], 
	[2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2 ]]) 
	
	
	start = time.time()
	theta_compute = optim_ik_UR5.optim_ik_UR5(theta_init, x_e_actual, y_e_actual, z_e_actual, R_e_actual, theta_min, theta_max)
	q.append(theta_compute)
	print(q)
	
	with open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/q_computed.csv', mode='w') as joint_angles:		
		joint_angles = csv.writer(joint_angles, delimiter=',')
		joint_angles.writerows(q)
	theta_init = theta_compute
	

	x_e_compute, y_e_compute, z_e_compute, R_e_compute = forward_kinem_UR5.forward_kinem_UR5(theta_compute)


	euler_actual = euler.mat2euler(R_e_actual, 'sxyz')
	euler_compute = euler.mat2euler(R_e_compute, 'sxyz')


	x_e_compute_array.append(x_e_compute)
	y_e_compute_array.append(y_e_compute)
	z_e_compute_array.append(z_e_compute)

	
	

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_end, y_end, z_end, label='desired parametric curve')
ax.plot(x_e_compute_array, y_e_compute_array, z_e_compute_array, label='parametric curve')
ax.legend()
plt.show()


