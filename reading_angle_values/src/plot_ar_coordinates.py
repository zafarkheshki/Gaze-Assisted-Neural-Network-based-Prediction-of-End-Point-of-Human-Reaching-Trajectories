
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

xfile = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_coordinates.txt')
x = []
y = []
z = []

for line in xfile:
    line = line.rstrip()
    wds = line.split()
    if wds[0] == 'x:':
        x.append(float(wds[1]))
    if wds[0] == 'y:':
        y.append(float(wds[1]))
    if wds[0] == 'z:':
        z.append(float(wds[1]))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(x)
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

