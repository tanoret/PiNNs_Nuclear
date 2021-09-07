import tensorflow as tf
import numpy as np
import time
from scipy.optimize import minimize

# Fourier feature projection class
class FourierFeatureProjection(tf.keras.layers.Layer):

    def __init__(self,
                 gaussian_projection: int,
                 gaussian_scale: float = 1.0,
                 trainable = False,
                 dim = 1,
                 weight_projection = True,
                 **kwargs):
        """
        Fourier Feature Projection layer from the paper:
        [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
        Add this layer immediately after the input layer.
        Args:
            gaussian_projection: Projection dimension for the gaussian kernel in fourier feature
                projection layer. Can be negative or positive integer.
                If <=0, uses identity matrix (basic projection) without gaussian kernel.
                If >=1, uses gaussian projection matrix of specified dim.
            gaussian_scale: Scale of the gaussian kernel in fourier feature projection layer.
                Note: If the scale is too small, convergence will slow down and obtain poor results.
                If the scale is too large (>50), convergence will be fast but results will be grainy.
                Try grid search for scales in the range [10 - 50].
        """
        super().__init__(**kwargs)

        if 'dtype' in kwargs:
            self._kernel_dtype = kwargs['dtype']
        else:
            self._kernel_dtype = tf.float32

        gaussian_projection = int(gaussian_projection)
        gaussian_scale = float(gaussian_scale)

        self.gauss_proj = gaussian_projection
        self.gauss_scale = gaussian_scale
        self.trainable = trainable
        self.dim = dim
        self.weight_projection = weight_projection

    def build(self, input_shape):
        # assume channel dim is always at last location
        input_dim = input_shape[-1]

        if self.gauss_proj <= 0:
            # Assume basic projection
            self.proj_kernel = tf.keras.layers.Dense(input_dim, use_bias=True, trainable=self.trainable,
                                                     kernel_initializer='identity', dtype=self._kernel_dtype,
                                                     bias_initializer='zeros')

        else:
            initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.gauss_scale)
            #initializer = tf.keras.initializers.Zeros()
            #initializer = tf.keras.initializers.Identity(gain=self.gauss_scale)
            self.proj_kernel = tf.keras.layers.Dense(self.gauss_proj, use_bias=True, trainable=self.trainable,
                                                     kernel_initializer=initializer, dtype=self._kernel_dtype,
                                                     bias_initializer='zeros')
            
        # Projection to input
        if ~self.trainable:
            initializer = tf.keras.initializers.Identity(gain=1)
            if self.weight_projection:
                self.proj_kernel_ident = tf.keras.layers.Dense(self.dim, use_bias=True, trainable=False,
                                                               kernel_initializer=initializer, dtype=self._kernel_dtype,
                                                               bias_initializer='zeros')
            else:
                self.proj_kernel_ident = tf.keras.layers.Dense(self.gauss_proj, use_bias=True, trainable=False,
                                                               kernel_initializer=initializer, dtype=self._kernel_dtype,
                                                               bias_initializer='zeros')

        self.built = True

    def call(self, inputs, **kwargs):
        
        # Direct part
        if self.trainable:
            x_proj_in = self.proj_kernel(inputs)
        else:
            x_proj_in = self.proj_kernel_ident(inputs)

        if self.weight_projection:
            x_input_shape_holder = tf.expand_dims(tf.reduce_sum(x_proj_in, axis=-1), axis=-1) * 0.0
        else:
            x_input_shape_holder = x_proj_in * 0.0

        if self.dim == 1:
          x_proj_bas = x_input_shape_holder + inputs
        if self.dim == 2:
          x_proj_bas_x = x_input_shape_holder * 0.0 + tf.expand_dims(inputs[..., 0], axis=-1)
          x_proj_bas_y = x_input_shape_holder * 0.0 + tf.expand_dims(inputs[..., 1], axis=-1)
          x_proj_bas = tf.concat([x_proj_bas_x, x_proj_bas_y], axis=-1)
        if self.dim == 3:
          x_proj_bas_x = x_input_shape_holder * 0.0 + tf.expand_dims(inputs[..., 0], axis=-1)
          x_proj_bas_y = x_input_shape_holder * 0.0 + tf.expand_dims(inputs[..., 1], axis=-1)
          x_proj_bas_z = x_input_shape_holder * 0.0 + tf.expand_dims(inputs[..., 2], axis=-1)
          x_proj_bas = tf.concat([x_proj_bas_x, x_proj_bas_y, x_proj_bas_z], axis=-1)

        #x_proj_bas = x_proj
        
        # Fourier part
        
        x_proj = 2.0 * np.pi * inputs
        x_proj = self.proj_kernel(x_proj)

        x_proj_sin = tf.sin(x_proj)
        x_proj_cos = tf.cos(x_proj)

        output = tf.concat([x_proj_bas, x_proj_sin, x_proj_cos], axis=-1)
        return output

    def get_config(self):
        config = {
            'gaussian_projection': self.gauss_proj,
            'gaussian_scale': self.gauss_scale
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
