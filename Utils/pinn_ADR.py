import tensorflow as tf
import numpy as np
import time
from scipy.optimize import minimize
from fourier_projection import FourierFeatureProjection
from lbfgs import lbfgs
from lbfgs import Struct


class PhysicsInformedNN_ADR(object):
  def __init__(self,
               layers,
               optimizer,
               logger,
               dim,
               points_dict,
               u_bc,
               bc_type = 'Dirichlet',
               kernel_projection='None',
               trainable_kernel=False,
               weight_projection=True):

    # Setting up dimension and handling colocation and boundary points
    self.dim = dim
    if dim == 1:
      x_eq = points_dict['x_eq']
      x_bc = points_dict['x_bc']
      X_f  = np.array([x_bc]).T
      X_eq = np.array([x_eq]).T
    elif dim == 2:
      x_eq, y_eq = points_dict['x_eq'], points_dict['y_eq']
      x_bc, y_bc = points_dict['x_bc'], points_dict['y_bc']
      X_f = np.concatenate([[x_bc, y_bc]], 1).T
      X_eq = np.concatenate([[x_eq, y_eq]], 1).T
    elif dim == 3:
      x_eq, y_eq, z_eq = points_dict['x_eq'], points_dict['y_eq'], points_dict['z_eq']
      x_bc, y_bc, z_bc = points_dict['x_bc'], points_dict['y_bc'],  points_dict['z_bc']
      X_f = np.concatenate([[x_bc, y_bc, z_bc]], 1).T
      X_eq = np.concatenate([[x_eq, y_eq, z_eq]], 1).T
    else:
      raise Exception('Dimension must be 1, 2, or 3')

    # Setting up input and output bounds for normalization
    ub, lb = X_f.max(), X_f.min()
    self.ub, self.lb = ub, lb
    self.kernel_projection = kernel_projection
    self.trainable_kernel = trainable_kernel
    self.gaussian_bound = layers[1]
    self.gaussian_scale = 1.
    self.layers = layers
    self.weight_projection = weight_projection

    # Descriptive Keras model [2, 20, …, 20, 1]
    self.restart_model()

    # Assign logegr, optimizer, and data type
    self.optimizer = optimizer
    self.logger = logger
    self.dtype = tf.float32

    # Separating the collocation coordinates and homogenizing data types
    if dim >= 1:
      self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=self.dtype)
      self.x_eq = tf.convert_to_tensor(X_eq[:, 0:1], dtype=self.dtype)
    if dim >= 2:
      self.y_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=self.dtype)
      self.y_eq = tf.convert_to_tensor(X_eq[:, 1:2], dtype=self.dtype)
    if dim >= 3:
      self.z_f = tf.convert_to_tensor(X_f[:, 2:3], dtype=self.dtype)
      self.z_eq = tf.convert_to_tensor(X_eq[:, 2:3], dtype=self.dtype)

    self.u_f = tf.convert_to_tensor(u_bc, dtype=self.dtype) # Note: u_f stores the normals if bc_type == Robin or Homogeneous
    self.X_f = X_f

    # Initialize Lagrange multipliers for homogeneous penalty
    self.lambda_bc, self.lambda_u = 1.,  1.
    self.scaling_penalty = len(x_eq)**2

    self.coupled_fields = {}
    self.residuals = {}
    self.residuals['advection'] = {}
    self.residuals['advection']['flag'] = False
    self.residuals['diffusion'] = {}
    self.residuals['diffusion']['flag'] = False
    self.residuals['diffusion_boundary'] = {}
    self.residuals['diffusion_boundary']['flag'] = False
    self.residuals['self_reaction'] = {}
    self.residuals['self_reaction']['flag'] = False
    self.residuals['external_reaction'] = {}

    # Assign boundary confition type
    self.bc_type = bc_type

  # Defining custom activation
  def swish(self, x):
    return x * tf.math.sigmoid(x)

  # Defining custom loss
  def __loss(self, lambda_bc = False, lambda_u = False):
    residual_internal = self.f_model()
    residual_boundary = self.return_bc_loss()
    lambda_bc = lambda_bc if lambda_bc else self.lambda_bc
    lambda_u  = lambda_u  if lambda_u  else self.lambda_u
    return  lambda_bc * tf.reduce_sum(tf.square(residual_boundary)) + \
            lambda_u * tf.reduce_sum(tf.square(residual_internal)) + \
            tf.constant(self.scaling_penalty, dtype=self.dtype) * \
            tf.square(
               tf.constant(lambda_bc, dtype=self.dtype) + \
               tf.constant(lambda_u, dtype=self.dtype) - \
               tf.constant(2.0, dtype=self.dtype)
              )

  def __grad(self):
    with tf.GradientTape() as tape:
      loss_value = self.__loss()
    return loss_value, tape.gradient(loss_value, self.__wrap_training_variables())

  def __wrap_training_variables(self):
    var = self.u_model.trainable_variables
    return var

  def add_coupled_variable(self, name, value):
    self.coupled_fields[name] = value

  def add_advection_term(self, velocity_name):
    self.residuals['advection']['flag'] = True
    self.residuals['advection']['velocity_name'] = velocity_name

  def add_diffusion_term(self, coef):
    self.residuals['diffusion']['flag'] = True
    self.residuals['diffusion']['coef'] = coef

  def add_boundary_diffusion_term(self, coef):
    self.residuals['diffusion_boundary']['flag'] = True
    self.residuals['diffusion_boundary']['coef'] = coef

  def add_self_rection_term(self, coef):
    self.residuals['self_reaction']['flag'] = True
    self.residuals['self_reaction']['coef'] = coef

  def add_external_reaction_term(self, other_field_name, coef, ID):
    self.residuals['external_reaction'][ID] = {}
    self.residuals['external_reaction'][ID]['other_field'] = other_field_name
    self.residuals['external_reaction'][ID]['coef'] = coef

  def nomalize_total_source_and_boundary(self):

    # Scaling boundary
    if self.bc_type == 'Dirichlet':
      self.residuals['mean_boundary'] = tf.reduce_mean(self.u_f) * 0.
      self.u_f = self.u_f - self.residuals['mean_boundary']
    else:
      self.residuals['mean_boundary'] = tf.constant(0., dtype=self.dtype)

    # Scaling internal source

    total_source = self.x_eq * tf.constant(0., dtype=self.dtype)

    if bool(self.residuals['external_reaction']):
      for index in self.residuals['external_reaction'].keys():
        other_field_name = self.residuals['external_reaction'][index]['other_field']
        if 'external' in other_field_name:
          total_source += self.residuals['external_reaction'][index]['coef'] * (self.coupled_fields[other_field_name] - self.residuals['mean_boundary'])
        else:
          total_source += self.residuals['external_reaction'][index]['coef'] * self.coupled_fields[other_field_name]

    else:
      total_source += tf.constant(1., dtype=self.dtype)

    if self.residuals['self_reaction']['flag']:
      #self.residuals['total_source_normalization'] = tf.reduce_mean(total_source) / tf.abs(self.residuals['self_reaction']['coef'])
      self.residuals['total_source_normalization'] = tf.abs(tf.reduce_mean(total_source) / tf.reduce_mean(tf.abs(self.residuals['self_reaction']['coef']) + 1.))
    else:
      self.residuals['total_source_normalization'] = tf.abs(tf.reduce_mean(total_source))

    if self.bc_type == 'Dirichlet': self.u_f /= self.residuals['total_source_normalization']


  ###########################################
  # Physics part
  ###########################################
  # The actual PINN
  def f_model(self):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime

    with tf.GradientTape(persistent=True) as tape:

      # Watching the two inputs we’ll need later, x and t
      if self.dim >= 1: tape.watch(self.x_eq)
      if self.dim >= 2: tape.watch(self.y_eq)
      if self.dim >= 3: tape.watch(self.z_eq)

      # Packing together the inputs
      if self.dim == 1: X_eq = tf.stack([self.x_eq[:,0]], axis=1)
      if self.dim == 2: X_eq = tf.stack([self.x_eq[:,0], self.y_eq[:,0]], axis=1)
      if self.dim == 3: X_eq = tf.stack([self.x_eq[:,0], self.y_eq[:,0], self.z_eq[:,0]], axis=1)

      # Getting the prediction
      u = self.u_model(X_eq)[..., 0]

      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      if self.dim >= 1: u_x = tape.gradient(u, self.x_eq)[..., 0]
      if self.dim >= 2: u_y = tape.gradient(u, self.y_eq)[..., 0]
      if self.dim >= 3: u_z = tape.gradient(u, self.z_eq)[..., 0]

    # Getting the other derivatives
    if self.dim >= 1: u_xx = tape.gradient(u_x, self.x_eq)[..., 0]
    if self.dim >= 2: u_yy = tape.gradient(u_y, self.y_eq)[..., 0]
    if self.dim >= 3: u_zz = tape.gradient(u_z, self.z_eq)[..., 0]

    # Letting the tape go
    del tape

    # Constructing residual
    residual = u * tf.constant(0.0, dtype=self.dtype) # Initialization trick - may improve later

    # Adding advection
    loc_residual = u * tf.constant(0.0, dtype=self.dtype) # Initialization trick - may improve later

    if self.residuals['advection']['flag']:
      velocity = self.coupled_fields[self.residuals['advection']['velocity_name']]
      if self.dim >= 1:
        x_velocity = tf.convert_to_tensor(velocity[:,0], dtype=self.dtype)
        loc_residual += x_velocity*u_x
      if self.dim >= 2:
        y_velocity = tf.convert_to_tensor(velocity[:,1], dtype=self.dtype)
        loc_residual += y_velocity*u_y
      if self.dim >= 3:
        z_velocity = tf.convert_to_tensor(velocity[:,2], dtype=self.dtype)
        loc_residual += z_velocity*u_z

      residual += loc_residual

    # Adding diffusion
    loc_residual = u * tf.constant(0.0, dtype=self.dtype)

    if self.residuals['diffusion']['flag']:
      if self.dim >= 1: loc_residual += u_xx
      if self.dim >= 2: loc_residual += u_yy
      if self.dim >= 3: loc_residual += u_zz
      residual -= self.residuals['diffusion']['coef'] * loc_residual

    # Adding self reaction term
    if self.residuals['self_reaction']['flag']:
      residual -= (self.residuals['self_reaction']['coef'] * u)

    # Adding external sources
    if bool(self.residuals['external_reaction']):
      for index in self.residuals['external_reaction'].keys():
        other_field_name = self.residuals['external_reaction'][index]['other_field']
        if 'external' in other_field_name:
          residual -= (self.residuals['external_reaction'][index]['coef'] * \
                       (self.coupled_fields[other_field_name] - self.residuals['mean_boundary'])) / self.residuals['total_source_normalization']
        else:
          residual -= (self.residuals['external_reaction'][index]['coef'] * self.coupled_fields[other_field_name]) / self.residuals['total_source_normalization']

    # Return the residul of the PiNN
    return residual


  def return_bc_loss(self):

    if self.bc_type == 'Dirichlet':
      # Get standard dirichlet loss
      u_pred = self.f_model_predict()
      return self.u_f - u_pred

    elif self.bc_type == 'Robin': # Extrapolated flux diffusion condition
      with tf.GradientTape(persistent=True) as tape:
        # Watching the two inputs we’ll need later, x and t
        if self.dim >= 1: tape.watch(self.x_f)
        if self.dim >= 2: tape.watch(self.y_f)
        if self.dim >= 3: tape.watch(self.z_f)
        # Packing together the inputs
        if self.dim == 1: X_f = tf.stack([self.x_f[:,0]], axis=1)
        if self.dim == 2: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0]], axis=1)
        if self.dim == 3: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0], self.z_f[:,0]], axis=1)
        # Getting the prediction
        u = self.u_model(X_f)[..., 0]

      # Deriving outside the tape
      if self.dim >= 1: u_x = tape.gradient(u, self.x_f)[..., 0]
      if self.dim >= 2: u_y = tape.gradient(u, self.y_f)[..., 0]
      if self.dim >= 3: u_z = tape.gradient(u, self.z_f)[..., 0]

      # Deleting tape
      del tape

      # Normal gradient
      if self.dim >= 1: u_ng = self.u_f * u_x
      if self.dim >= 2: u_ng += self.u_f * u_y
      if self.dim >= 3: u_ng += self.u_f * u_z

      return u + 2*self.residuals['diffusion_boundary']['coef']*u_ng

    elif self.bc_type == 'Homogeneous':
      with tf.GradientTape(persistent=True) as tape:
        # Watching the two inputs we’ll need later, x and t
        if self.dim >= 1: tape.watch(self.x_f)
        if self.dim >= 2: tape.watch(self.y_f)
        if self.dim >= 3: tape.watch(self.z_f)
        # Packing together the inputs
        if self.dim == 1: X_f = tf.stack([self.x_f[:,0]], axis=1)
        if self.dim == 2: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0]], axis=1)
        if self.dim == 3: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0], self.z_f[:,0]], axis=1)
        # Getting the prediction
        u = self.u_model(X_f)[..., 0]

      # Deriving outside the tape
      if self.dim >= 1: u_x = tape.gradient(u, self.x_f)[..., 0]
      if self.dim >= 2: u_y = tape.gradient(u, self.y_f)[..., 0]
      if self.dim >= 3: u_z = tape.gradient(u, self.z_f)[..., 0]

      # Deleting tape
      del tape

      # Normal gradient
      if self.dim >= 1: u_ng = self.u_f * u_x
      if self.dim >= 2: u_ng += self.u_f * u_y
      if self.dim >= 3: u_ng += self.u_f * u_z

      return u_ng


  def get_weights(self):
    w = []
    if self.kernel_projection == 'Gaussian':
      ns = 2
    elif self.kernel_projection == 'Fourier':
      ns = 1 if self.trainable_kernel else 2
    else:
      ns = 1
    for layer in self.u_model.layers[ns:]:
      weights_biases = layer.get_weights()
      weights = weights_biases[0].flatten()
      biases = weights_biases[1]
      w.extend(weights)
      w.extend(biases)
    return tf.convert_to_tensor(w, dtype=self.dtype)

  def set_weights(self, w):
    if self.kernel_projection == 'Gaussian':
      ns = 2
    elif self.kernel_projection == 'Fourier':
      ns = 1 if self.trainable_kernel else 2
    else:
      ns = 1
    for i, layer in enumerate(self.u_model.layers[ns:]):
      start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
      end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
      weights = w[start_weights:end_weights]
      w_div = int(self.sizes_w[i] / self.sizes_b[i])
      weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
      biases = w[end_weights:end_weights + self.sizes_b[i]]
      weights_biases = [weights, biases]
      layer.set_weights(weights_biases)

  def summary(self):
    return self.u_model.summary()

  def return_numpy_loss(self, lambda_array):
    lambda_bc, lambda_u = lambda_array
    return self.__loss(lambda_bc=lambda_bc, lambda_u=lambda_u).numpy()

  def __return_jacobian(self, lambda_array):
    return ndiff.Jacobian(lambda x: self.return_numpy_loss(x))(lambda_array).ravel()

  # The training function
  def fit(self,
          tf_epochs=5000,
          coupled_optimizer={'nt_config': Struct(), 'batches': 0},
          restart_tf = True):

    self.nomalize_total_source_and_boundary()
    self.restart_model()

    def loss_and_flat_grad(w):
      with tf.GradientTape() as tape:
        self.set_weights(w)
        loss_value = self.__loss()
      grad = tape.gradient(loss_value, self.u_model.trainable_variables)
      grad_flat = []
      for g in grad:
        grad_flat.append(tf.reshape(g, [-1]))
      grad_flat =  tf.concat(grad_flat, 0)
      return loss_value, grad_flat

    self.logger.log_train_start(self)

    for n in range(coupled_optimizer['batches']):

      self.restart_model()

      if restart_tf or n == 0:

        self.logger.log_train_opt("Adam")
        for epoch in range(tf_epochs):
          # Optimization step
          loss_value, grads = self.__grad()
          self.optimizer.apply_gradients(zip(grads, self.__wrap_training_variables()))
          self.logger.log_train_epoch(epoch, loss_value)

        self.logger.log_train_opt("LBFGS")

      print('Working on batch {0}'.format(n))

      lbfgs(loss_and_flat_grad,
        self.get_weights(),
        coupled_optimizer['nt_config'], Struct(), True,
        lambda epoch, loss, is_iter:
          self.logger.log_train_epoch(epoch, loss, "", is_iter))

      #print('Optimizing Lagrange multipliers')
      bnds = ((0.5, 2.), (0.5, 2.))
      x0 = np.array([self.lambda_bc, self.lambda_u]).copy()
      res = minimize(self.return_numpy_loss, x0, method='L-BFGS-B', #jac=self.__return_jacobian,
                     tol=1e-6, bounds=bnds, options={'maxiter': 1, 'eps': 1e-10})
      #print(res)
      [self.lambda_bc, self.lambda_u] = res.x
      #print(np.array([self.lambda_bc, self.lambda_u]))

    self.logger.log_train_end(tf_epochs + coupled_optimizer['nt_config'].maxIter)


  def f_model_predict(self, new_pred=False, X_u=[], scale=True):
    # Using the new GradientTape paradigm of TF2.0,
    # which keeps track of operations to get the gradient at runtime

    if new_pred:

      if self.dim >= 1: x_loc = tf.convert_to_tensor(X_u[...,0], dtype=self.dtype)
      if self.dim >= 2: y_loc = tf.convert_to_tensor(X_u[...,1], dtype=self.dtype)
      if self.dim >= 3: z_loc = tf.convert_to_tensor(X_u[...,2], dtype=self.dtype)

      # Creating prediction array
      if self.dim >= 1: X_u = tf.stack([x_loc], axis=1)
      if self.dim >= 2: X_u = tf.stack([x_loc, y_loc], axis=1)
      if self.dim >= 3: X_u = tf.stack([x_loc, y_loc, z_loc], axis=1)

      # Getting the prediction
      if scale:
        u = self.u_model(X_u)[...,0] * self.residuals['total_source_normalization'] + self.residuals['mean_boundary']
      else:
        u = self.u_model(X_u)[...,0]

      # Buidling the PINNs
      return u

    else:

      # Packing together the inputs
      if self.dim == 1: X_f = tf.stack([self.x_f[:,0]], axis=1)
      if self.dim == 2: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0]], axis=1)
      if self.dim == 3: X_f = tf.stack([self.x_f[:,0], self.y_f[:,0], self.z_f[:,0]], axis=1)

      # Getting the prediction
      u = self.u_model(X_f)[...,0]

      # Buidling the PINNs
      return u


  def predict(self, X_u, scale=True):

    X_u = tf.convert_to_tensor(X_u, dtype=self.dtype)
    u = self.f_model_predict(new_pred=True, X_u=X_u, scale=scale)

    return u

  def return_loss(self):
    return self.__loss()

  def save_model(self, name):
    self.u_model.save(name)

  def load_model(self, name):
    self.u_model = tf.keras.models.load_model(name)


  def restart_model(self):

    layers = self.layers

    #self.gaussian_bound = 2**min(int(np.sqrt(layers[1])+1),12)
    self.u_model = tf.keras.Sequential()
    self.u_model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
    # Adding projection kernel if defined
    if self.kernel_projection == 'Gaussian':
      self.u_model.add(tf.keras.layers.Lambda(lambda X: (X - self.lb)/(self.ub - self.lb)))
      self.u_model.add(tf.keras.layers.experimental.RandomFourierFeatures(
          output_dim=self.gaussian_bound, kernel_initializer='gaussian', scale=1.))
    elif self.kernel_projection == 'Fourier':
      self.u_model.add(tf.keras.layers.Lambda(lambda X: (X - self.lb)/(self.ub - self.lb)))
      self.u_model.add(FourierFeatureProjection(
                  gaussian_projection=self.gaussian_bound ,
                  gaussian_scale=self.gaussian_scale, trainable=self.trainable_kernel, 
                  dim=self.dim, weight_projection=self.weight_projection))
    else:
      self.u_model.add(tf.keras.layers.Lambda(lambda X: 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0))
    for width in layers[1:-1]:
        self.u_model.add(tf.keras.layers.Dense(
          width, activation=tf.nn.tanh,
          kernel_initializer='glorot_normal'))
    self.u_model.add(tf.keras.layers.Dense(
          layers[-1], activation='swish',
          kernel_initializer='glorot_normal'))

    # Computing the sizes of weights/biases for future decomposition
    self.sizes_w = []
    self.sizes_b = []
    for i, width in enumerate(layers):
      if i != 1:
        if self.kernel_projection == 'Fourier' and i==0:
          if self.trainable_kernel:
            self.sizes_w.append(int(self.gaussian_bound * self.dim))
            self.sizes_b.append(int(self.gaussian_bound))
          if self.weight_projection:
              self.sizes_w.append(int(layers[1] * (2 * self.gaussian_bound + self.dim)))
          else:
              self.sizes_w.append(int(layers[1] * (2 + self.dim) * self.gaussian_bound ))
          self.sizes_b.append(int(layers[1]))
        elif self.kernel_projection == 'Gaussian' and i==0:
          self.sizes_w.append(int(layers[1] * self.gaussian_bound))
          self.sizes_b.append(int(layers[1]))
        else:
          self.sizes_w.append(int(width * layers[1]))
          self.sizes_b.append(int(width if i != 0 else layers[1]))
