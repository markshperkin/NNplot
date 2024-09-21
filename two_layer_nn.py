import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the weight matrices and biases
V_T = np.array([[-2.69, -2.80], [-3.39, -4.56]])
b_v = np.array([-2.21, 4.76])
W = np.array([-4.91, 4.95])
b_w = -2.28

# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    return np.where(z >= 0, 1, 0)

def radial_basis_function(z):
    return np.exp(-z**2)

# Define the two-layer neural network function
def two_layer_nn(x1, x2, activation_func):
    x = np.stack([x1, x2], axis=-1)  
    z = np.dot(x, V_T.T) + b_v 
    a = activation_func(z)
    y = np.dot(a, W) + b_w
    return y

# Define input domain
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
x1, x2 = np.meshgrid(x1, x2)

# Calculate outputs for each activation function
y_sigmoid = two_layer_nn(x1, x2, sigmoid)
y_hard_limit = two_layer_nn(x1, x2, hard_limit)
y_rbf = two_layer_nn(x1, x2, radial_basis_function)

# Create 3 plots
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x1, x2, y_sigmoid, cmap='Reds')
ax1.set_title("Sigmoid Activation Function")
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('y')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x1, x2, y_hard_limit, cmap='Blues')
ax2.set_title("Hard Limit Activation Function")
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x1, x2, y_rbf, cmap='Greens')
ax3.set_title("Radial Basis Function")
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')

plt.tight_layout()
plt.show()
