

from numpy import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import forward_kinem_UR5

theta_temp = loadmat('theta_UR5.mat')
theta = theta_temp['theta']

delt = 0.04
num = shape(theta)[0]

x_e = zeros(num)
y_e = zeros(num)
z_e = zeros(num)



for i in range(0,  num):
	x_e[i], y_e[i], z_e[i], R_fin = forward_kinem_UR5.forward_kinem_UR5(theta[i])


theta_1_dot = diff(theta[:,0])/delt
theta_2_dot = diff(theta[:,1])/delt
theta_3_dot = diff(theta[:,2])/delt
theta_4_dot = diff(theta[:,3])/delt
theta_5_dot = diff(theta[:,4])/delt
theta_6_dot = diff(theta[:,5])/delt
# theta_7_dot = diff(theta[:,6])/delt



x_e_dot = diff(x_e)/delt
y_e_dot = diff(y_e)/delt
z_e_dot = diff(z_e)/delt


plt.plot(theta_6_dot)
plt.show()


