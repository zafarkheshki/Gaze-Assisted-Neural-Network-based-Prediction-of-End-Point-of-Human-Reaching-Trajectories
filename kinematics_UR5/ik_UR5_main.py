

from numpy import *
import matplotlib.pyplot as plt 
import forward_kinem_UR5
import optim_ik_UR5
from transforms3d import *
import time




theta_min = array([-165, -100, -165, -165, -165, -1.0, -165  ])*pi/180
theta_max = array([ 165,   101,  165,  1.0,    165, 214, 165  ])*pi/180 

theta_actual = random.uniform(theta_min, theta_max, 7)

x_e_actual, y_e_actual, z_e_actual, R_e_actual = forward_kinem_UR5.forward_kinem_UR5(theta_actual)

theta_init = (theta_min+theta_max)

start = time.time()
theta_compute = optim_ik_UR5.optim_ik_UR5(theta_init, x_e_actual, y_e_actual, z_e_actual, R_e_actual, theta_min, theta_max)
print time.time()-start

print theta_compute
print theta_actual

x_e_compute, y_e_compute, z_e_compute, R_e_compute = forward_kinem_UR5.forward_kinem_UR5(theta_compute)

print x_e_actual, x_e_compute
print y_e_actual, y_e_compute
print z_e_actual, z_e_compute

euler_actual = euler.mat2euler(R_e_actual, 'sxyz')
euler_compute = euler.mat2euler(R_e_compute, 'sxyz')

print euler_actual
print euler_compute







