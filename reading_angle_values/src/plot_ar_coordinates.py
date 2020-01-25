
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xfile = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_coordinates.txt')
x = []
y = []
z = []

for line in xfile:
    line = line.rstrip()
    wds = line.split()
    if wds[0] == 'x' and wds[1] != 'nan':
        x.append(float(wds[1]))
    if wds[0] == 'y' and wds[1] != 'nan':
        y.append(float(wds[1]))
    if wds[0] == 'z' and wds[1] != 'nan':
        z.append(float(wds[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

