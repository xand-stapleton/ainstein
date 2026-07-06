"""
DiffeomorphismNet: a residual MLP approximating a smooth map phi: R^4 -> R^4,
initialised near the identity (phi(x) = x + delta_phi(x), delta_phi ~ 0).

Used to find an explicit coordinate diffeomorphism between the learned Ricci-flat
metric and the analytic Schwarzschild metric, by minimising the pullback loss

    L_pullback = || J_phi^T  g_an(phi(x))  J_phi  -  g_pred(x) ||_F^2 / ||g_pred(x)||_F^2

where J_phi[i,mu] = d phi^i / d x^mu  is the 4x4 Jacobian computed via autograd.
"""

from __future__ import annotations

import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")


@tfk.utils.register_keras_serializable()
class DiffeomorphismNet(tfk.Model):
    """Residual MLP representing phi(x) = x + delta_phi_theta(x).

    Parameters
    ----------
    n_dim        : dimension of the manifold (4 for Schwarzschild R^2 x S^2).
    hidden_units : width of each hidden layer.
    hidden_depth : number of hidden layers.
    init_scale   : std of the output layer's kernel initialisation; controls how
                   far from the identity the map starts.  Keep small (1e-3).
    """

    def __init__(
        self,
        n_dim: int = 4,
        hidden_units: int = 256,
        hidden_depth: int = 4,
        init_scale: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.hidden_units = hidden_units
        self.hidden_depth = hidden_depth
        self.init_scale = init_scale

        self._hidden = [
            tfk.layers.Dense(
                hidden_units,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                dtype=tf.float64,
            )
            for _ in range(hidden_depth)
        ]
        # Output layer: TruncatedNormal with tiny std so phi ≈ id at start.
        self._output = tfk.layers.Dense(
            n_dim,
            kernel_initializer=tfk.initializers.TruncatedNormal(stddev=init_scale),
            bias_initializer="zeros",
            dtype=tf.float64,
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass: phi(x) = x + delta(x).

        Args:
            x : tf.Tensor (batch, n_dim)
        Returns:
            phi(x) : tf.Tensor (batch, n_dim)
        """
        h = x
        for layer in self._hidden:
            h = layer(h)
        return x + self._output(h)

    def jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the Jacobian J[s, i, mu] = d phi^i / d x^mu via GradientTape.

        Args:
            x : tf.Tensor (batch, n_dim)  — must be watched or be a variable.
        Returns:
            J : tf.Tensor (batch, n_dim, n_dim)
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            phi = self.call(x)
        return tape.batch_jacobian(phi, x)  # (batch, n_dim, n_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            n_dim=self.n_dim,
            hidden_units=self.hidden_units,
            hidden_depth=self.hidden_depth,
            init_scale=self.init_scale,
        )
        return cfg
