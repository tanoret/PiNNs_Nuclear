import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from scipy.optimize import minimize
from fourier_projection import FourierFeatureProjection
from pinn_ADR import PhysicsInformedNN_ADR
from logger import Logger
from lbfgs import lbfgs
from lbfgs import Struct

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(17)
tf.random.set_seed(17)

#############################
# Parameters
#############################
tf_epochs = 200 # Number of training epochs for the regression part
nt_epochs = 1000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 100 # Number of samples per boundary
N_internal = 5000 # Number of samples per boundary

#############################
# Network srchitecture
#############################
dim = 2
layers = [dim] + 7*[30] + [1]

#################################
# Setting logger and optimizer
#################################
logger = Logger(frequency=20)
def error():
  return tf.reduce_sum((tf.square(pinn_ADR.return_bc_loss()))).numpy()
logger.set_error_fn(error)

#################################
# Setting up tf optimizer
#################################
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

coupled_optimizer = {}
coupled_optimizer['nt_config'] = Struct()
coupled_optimizer['nt_config'].learningRate = 0.8
coupled_optimizer['nt_config'].maxIter = nt_epochs
coupled_optimizer['nt_config'].nCorrection = 50
coupled_optimizer['nt_config'].tolFun = 1.0 * np.finfo(float).eps
coupled_optimizer['batches'] = n_batches

##############################
# Domain
##############################
x_min, x_max = 0., 1.
y_min, y_max = 0., 1.

# Boundaries
delbnd = 0.005
bottom = (np.linspace(x_min+delbnd,x_max-delbnd,N_boundary), np.linspace(y_min,y_min,N_boundary))
top    = (np.linspace(x_min+delbnd,x_max-delbnd,N_boundary), np.linspace(y_max,y_max,N_boundary))
left   = (np.linspace(x_min,x_min,N_boundary), np.linspace(y_min+delbnd,y_max-delbnd,N_boundary))
right  = (np.linspace(x_max,x_max,N_boundary), np.linspace(y_min+delbnd,y_max-delbnd,N_boundary))

x_cord_bc = np.array([bottom[0], top[0], left[0], right[0]])
y_cord_bc = np.array([bottom[1], top[1], left[1], right[1]])

# Boundary conditions
left_src = left[0]*0.
left_src[(left[1] < 0.6) & (left[1] > 0.4)] = 0.
u_bc = np.array([bottom[0]*0., top[0]*0., left_src, right[0]*0.])

# Internal points
n_points = int(np.sqrt(N_internal))
delta_x, delta_y = x_max/n_points, y_max/n_points
nx, ny = (n_points, n_points)
nd = 4.0
x = np.linspace(0 + delta_x/nd, x_max - delta_x/nd, nx)
y = np.linspace(0 + delta_x/nd, y_max - delta_x/nd, ny)
x_grid, y_grid = np.meshgrid(x, y)
x_cord_int, y_cord_int = x_grid, y_grid
#x_cord_int = np.random.uniform(low=x_min, high=x_max, size=(N_internal,))
#y_cord_int = np.random.uniform(low=y_min, high=y_max, size=(N_internal,))

# Training data
points_dict = {}
points_dict['x_bc'] = x_cord_bc.flatten()
points_dict['y_bc'] = y_cord_bc.flatten()
points_dict['x_eq'] = x_cord_int.flatten()
points_dict['y_eq'] = y_cord_int.flatten()

u_train_bc = u_bc.flatten()

#####################
# Creating PiNNS
#####################

# Creating PiNNs class
pinn_ADR = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger, 
                                 dim = dim, points_dict=points_dict, 
                                 u_bc=u_train_bc, bc_type = 'Dirichlet',
                                 kernel_projection='Fourier',
                                 trainable_kernel=False,
                                 weight_projection=True)

pinn_ADR.gaussian_bound = 128 #layers[1]//2
pinn_ADR.gaussian_scale = 1.


# Adding advection
velocity = np.concatenate([[points_dict['x_eq']*0.+0.01, points_dict['y_eq']*0.+0.01]], 1).T
pinn_ADR.add_coupled_variable('velocity', velocity)
pinn_ADR.add_advection_term('velocity')

# Adding diffusion
diffusivity = points_dict['x_eq'] * 0. + 1e-2
cond_x = (points_dict['x_eq'] > 0.5)
diffusivity[cond_x] = 1.0
pinn_ADR.add_diffusion_term(diffusivity)

# Adding Power
power = tf.constant(0.5, dtype=pinn_ADR.dtype)
pinn_ADR.add_coupled_variable('power', power)
pinn_ADR.add_external_reaction_term('power', coef=1.0, ID=0)

# Adding volumetric cooling
self_reaction_coef = points_dict['x_eq'] * 0.
cond_x = (points_dict['x_eq'] > 0.3) & (points_dict['x_eq'] < 0.5)
cond_y = (points_dict['y_eq'] > 0.7) & (points_dict['y_eq'] < 0.9)
self_reaction_coef[cond_x & cond_y] = -5.
self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=pinn_ADR.dtype)
pinn_ADR.add_self_rection_term(self_reaction_coef)
# pinn_ADR.add_coupled_variable('T_external', x_train_int * 0.)
# pinn_ADR.add_external_reaction_term('T_external', coef=self_reaction_coef, ID=1)


# Training
pinn_ADR.fit(tf_epochs, coupled_optimizer)

################################
# Plot
################################

nx, ny = (60, 60)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
x_grid, y_grid = np.meshgrid(x, y)
x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
X_u = np.concatenate([[x_grid, y_grid]], 1).T

conc = pinn_ADR.predict(X_u)

plt.figure(figsize=[6,6])
plt.tricontourf(x_grid, y_grid, conc, levels=30)
plt.title('Solution')
plt.colorbar()
plt.show()


