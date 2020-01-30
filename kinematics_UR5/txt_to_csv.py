import csv
import itertools

xfile = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.txt')

lines = xfile.readline()

for i in range(len(lines)):
    if lines.strip() == 'position:':
        lines = xfile.readline()
    if lines.strip() == 'orientation:':
        lines = xfile.readline()
    # continue
    
grouped = itertools.izip(*[lines] * 7)
with open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.csv', 'w') as out_file:
    writer = csv.writer(out_file)
    # writer.writerow(('title', 'intro'))
    writer.writerows(grouped)