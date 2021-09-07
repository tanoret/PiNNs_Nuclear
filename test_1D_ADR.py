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
np.random.seed(1234)
tf.random.set_seed(1234)

#############################
# Parameters
#############################
tf_epochs = 200 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 100 # Number of samples per boundary
N_internal = 100 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 4*[30] + [1]

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
coupled_optimizer['nt_config'].learningRate = 0.5
coupled_optimizer['nt_config'].maxIter = nt_epochs
coupled_optimizer['nt_config'].nCorrection = 50
coupled_optimizer['nt_config'].tolFun = 1.0 * np.finfo(float).eps
coupled_optimizer['batches'] = n_batches

##############################
# Domain
##############################
x_min, x_max = 0., 1.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)
x_cord_bc = np.array([left, right])

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc = np.array([left*0.+0., right*0.+1.])

# Internal points
n_points = N_internal
delta_x = x_max/n_points
nd = 4.0
x = np.linspace(x_min + delta_x/nd, x_max - delta_x/nd, n_points)

# Training data
points_dict = {}
points_dict['x_bc'] = x_cord_bc.flatten()
points_dict['x_eq'] = x.flatten()

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

pinn_ADR.gaussian_bound = 512 #2**3
pinn_ADR.gaussian_scale = 1.

# Adding advection
# velocity = np.concatenate([[points_dict['x_eq']*0.pinp+0.1]], 1).T
# pinn_ADR.add_coupled_variable('velocity', velocity)
# pinn_ADR.add_advection_term('velocity')

# Adding diffusion
#diffusivity = 0.1
diffusivity = x * 0 + 10.
diffusivity[(x > 0.) & (x <= 0.3)] = 1.
diffusivity[(x > 0.3) & (x <= 0.7)] = .1
diffusivity[(x > 0.7) & (x <= 1.0)] = 1.
diffusivity = tf.convert_to_tensor(diffusivity, dtype=pinn_ADR.dtype)
pinn_ADR.add_diffusion_term(diffusivity)

# Adding Power
power = tf.constant(1.0, dtype=pinn_ADR.dtype)
pinn_ADR.add_coupled_variable('power', power)
pinn_ADR.add_external_reaction_term('power', coef=1.0, ID=0)

# Adding volumetric cooling
self_reaction_coef = x * 0.
self_reaction_coef[(x > 0.5) & (x < 0.8)] = -10.
self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=pinn_ADR.dtype)
pinn_ADR.add_self_rection_term(self_reaction_coef)
# pinn_ADR.add_coupled_variable('T_external', x_train_int * 0.)
# pinn_ADR.add_external_reaction_term('T_external', coef=self_reaction_coef, ID=1)

# Training
pinn_ADR.fit(tf_epochs, coupled_optimizer)

######################################
# Plot
######################################
sol = pinn_ADR.predict(np.array([x]).T)

plt.figure(figsize=[6,5])
plt.plot(x, pinn_ADR.predict(np.array([x]).T))
plt.xlabel('x')
plt.ylabel('$\phi$')
plt.grid()
