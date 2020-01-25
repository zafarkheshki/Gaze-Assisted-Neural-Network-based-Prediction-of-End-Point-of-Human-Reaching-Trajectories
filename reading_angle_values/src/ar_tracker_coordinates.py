
xfile = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_tracker_pose.txt')
line = xfile.readline()
counter = 3

while line:
    if line.strip() == 'position:':
        line = xfile.readline()
    if counter == 0:
        line = xfile.readline()
        line = xfile.readline()
        line = xfile.readline()
        line = xfile.readline()
        line = xfile.readline()
        counter = 3
        continue

    with open ('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/ar_coordinates.txt', 'a+') as text_file:
        text_file.write(line)

    print (line)
    line = xfile.readline()
    counter = counter - 1



