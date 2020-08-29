import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import random
from scipy.interpolate import CubicSpline
from numpy import hstack

# Downlown sampler
def down_sampler(arr):
    arr = np.array(arr)
    x = random.sample(range(0,len(arr)), 20)
    x.sort()
    arr_sampled = arr[x]
    arr_sampled[0] = arr[0]
    arr_sampled[len(arr_sampled)-1] = arr[len(arr)-1]
    
    return arr_sampled

# Input data
x = []
y = []
z = []

input_data = []
for i in range(330):
    trajectory_points = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/New_Data/AR_tracker_new/ar_tracker_pose'+str(i)+'.txt')
    lines = trajectory_points.readline()
    x_p = []
    y_p = []
    z_p = []
    counter = 3
    
    while lines:
        if lines.strip() == 'position:':
            lines = trajectory_points.readline()
        if counter == 0:
            lines = trajectory_points.readline()
            lines = trajectory_points.readline()
            lines = trajectory_points.readline()
            lines = trajectory_points.readline()
            lines = trajectory_points.readline()
            counter = 3
            continue

        wds = lines.split()
        if wds[0] == 'x:':
            x_p.append(float(wds[1]))
        if wds[0] == 'y:':
            y_p.append(float(wds[1]))
        if wds[0] == 'z:':
            z_p.append(float(wds[1]))    
            
        lines = trajectory_points.readline()
        counter = counter-1 

    traj_points = np.column_stack((x_p, y_p, z_p))
    points_sampled = down_sampler(traj_points)
    
    x_p = points_sampled[:,0]
    y_p = points_sampled[:,1]
    z_p = points_sampled[:,2]
    # print(i)
    a = np.arange(len(x_p))
    csx = CubicSpline(a, x_p)
    csy = CubicSpline(a, y_p)
    csz = CubicSpline(a, z_p)
    xs = np.arange(0, a[-1], 0.1)
    x_p = csx(xs)
    y_p = csy(xs)
    z_p = csz(xs)


    input_data.append(x_p)
    input_data.append(y_p)
    input_data.append(z_p)

input_data = np.concatenate(input_data)
input_data = input_data.reshape(330,570).astype('float64')
train_input = input_data[:300,:].astype('float64')
test_input = input_data[300:,:].astype('float64')
# print(input_data.shape)


# Output data
e_x = []
e_y = []
e_z = []
end_eff_points = open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/New_Data/End_eff_pos/end_effector.txt')
position = end_eff_points.readline()

while len(position.split())>0:
    pts = position.split()
    # print(pts)
    e_x.append(float(pts[0]))
    e_y.append(float(pts[1]))
    e_z.append(float(pts[2]))

    position = end_eff_points.readline()

end_effector = np.column_stack((e_x, e_y, e_z)).astype('float64')
train_output = end_effector[:300,:].astype('float64')
test_output = end_effector[300:,:].astype('float64')

# Model
class MLP(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(570,350),
            torch.nn.BatchNorm1d(350),
            torch.nn.Tanh(),
            torch.nn.Linear(350,200),
            torch.nn.BatchNorm1d(200),
            torch.nn.Tanh(),
            torch.nn.Linear(200,75),
            torch.nn.BatchNorm1d(75),
            torch.nn.Tanh(),
            torch.nn.Linear(75,25),
            torch.nn.BatchNorm1d(25),
            torch.nn.Tanh(),
            torch.nn.Linear(25,5),
            torch.nn.BatchNorm1d(5),
            torch.nn.Tanh(),
            torch.nn.Linear(5,3)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        out = self.layers(x)
        return out
inputDim = 570
outputDim = 3
learningRate = 0.0025
epochs = 1500

model = MLP(inputDim, outputDim)
model.double()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)


# Training
train_losses = []
valid_losses = []
for epoch in range(epochs):
    # Converting inputs and labels to Variable
    
    inputs = Variable(torch.from_numpy(train_input)).double()
    labels = Variable(torch.from_numpy(train_output)).double()
    
    model.train()
    
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs).double()

    # print(outputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels).double()
        
    # print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    train_losses.append(loss.item())
    
    model.eval()

    with torch.no_grad():
        outputs = model(inputs).double()
        loss = criterion(outputs, labels)
        valid_losses.append(loss.item())

    print('epoch {}, train_loss {}, valid_loss {}'.format(epoch, np.mean(train_losses), np.mean(valid_losses)))


torch.save(model.state_dict(), os.path.join(os.getcwd(),'traj_ee_model.pt'))


# Predicting
with torch.no_grad(): # we don't need gradients in the testing phase
    
    test_predicted = model(Variable(torch.from_numpy(test_input))).data.numpy()


with torch.no_grad(): # we don't need gradients in the testing phase
    
    train_predicted = model(Variable(torch.from_numpy(train_input))).data.numpy()

# Saving the predicted values in a text file
traj_ee_results = []
traj_ee_results.append(train_predicted)
traj_ee_results.append(test_predicted)
traj_ee_results = np.concatenate(traj_ee_results)
with open('/home/zafar/catkin_ws/src/Thesis/reading_angle_values/New_Data/traj_ee_results.txt', 'w') as text_file:
# with open('your_file.txt', 'w') as f:
    for item in traj_ee_results:
        text_file.write("%s\n" %item)


list_diff = []
for a in range(len(train_predicted)):
    b = math.sqrt((train_output[a,0]-train_predicted[a,0])**2+(train_output[a,1]-train_predicted[a,1])**2+(train_output[a,2]-train_predicted[a,2])**2)
    list_diff.append(b)

print('Training Errors')
print(max(list_diff))
print(np.mean(list_diff))


list_diff_test = []
for a in range(len(test_predicted)):
    b = math.sqrt((test_output[a,0]-test_predicted[a,0])**2+(test_output[a,1]-test_predicted[a,1])**2+(test_output[a,2]-test_predicted[a,2])**2)
    list_diff_test.append(b)

print('Test Errors')
print(max(list_diff_test))
print(np.mean(list_diff_test))


# Plotting
fig = plt.figure("train_prediction")
ax = fig.add_subplot(111, projection='3d')
actual_train = ax.scatter(train_output[:,0], train_output[:,1], train_output[:,2], 'b', label='Actual')
predicted_train = ax.scatter(train_predicted[:,0], train_predicted[:,1], train_predicted[:,2], 'rs', label='Predicted')
# ax.scatter(test_output[:,0], test_output[:,1], test_output[:,2], 'b')
# ax.scatter(test_predicted[:,0], test_predicted[:,1], test_predicted[:,2], 'rs')
ax.set_xlabel('X-axis (m)')
ax.set_ylabel('Y-axis (m)')
ax.set_zlabel('Z-axis (m)')
ax.legend([actual_train, predicted_train], ['Actual', 'Predicted'])


fig = plt.figure("test_prediction")
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_output[:,0], train_output[:,1], train_output[:,2], 'b')
# ax.scatter(train_predicted[:,0], train_predicted[:,1], train_predicted[:,2], 'rs')
actual_test = ax.scatter(test_output[:,0], test_output[:,1], test_output[:,2], 'b', label='Actual')
predicted_test = ax.scatter(test_predicted[:,0], test_predicted[:,1], test_predicted[:,2], 'rs', label='Predicted')
ax.set_xlabel('X-axis (m)')
ax.set_ylabel('Y-axis (m)')
ax.set_zlabel('Z-axis (m)')
ax.legend([actual_test, predicted_test], ['Actual', 'Predicted'])


fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
ax1.plot(train_losses, label='train')
ax1.plot(valid_losses, label='valid')
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='best')
plt.show()

