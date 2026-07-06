from __future__ import annotations

import tensorflow as tf

tfk = tf.keras
tfk.backend.set_floatx("float64")
from configs.base import BaseConfig
from configs.schwarzschild import SchwarzschildConfig
from geometry.schwarzschild import embed_S2_coords
from losses.schwarzschild import (TotalSchwarzschildLocal2DLoss,
                                  TotalSchwarzschildLoss)
from network.base import (BaseGlobalModel, BaseNetwork, BasePatchSubmodel,
                          SerialisableModel)


@tfk.utils.register_keras_serializable()
class SchwarzschildPatchSubModel(BasePatchSubmodel):
    # This will also inherit from the SerialisableModel
    def __init__(self, config, n_out, input_dim=None, **kwargs):
        # For the Lorentzian Schwarzschild 5D ambient metric, L[0,0] sits at
        # index 5 in the 15-component Cholesky vector (fill_triangular convention,
        # empirically verified).  G[0,0] = -L[0,0]^2, so raw=0 → G[0,0]=0
        # (degenerate) at zero-centred initialisation.  Initialising the output
        # bias with +1 at index 5 maps raw=0 → L[0,0]=1 → G[0,0]=-1 at init,
        # breaking the saddle.  The bias is trainable, so the network learns the
        # true output rather than correcting a permanent offset.
        output_bias_init = None
        if getattr(config.model_specific, "lorentzian", False):
            import numpy as np
            import tensorflow_probability as tfp

            n_mat = int(round((-1 + np.sqrt(1 + 8 * n_out)) / 2))  # 5 for n_out=15
            # Bias = fill_triangular_inverse(I_n): the unique vector v s.t.
            # fill_triangular(v) = I_{n_mat}.  At zero-weight init this gives
            # G_AB = eta_{n_mat} (Lorentzian identity), so every diagonal of L
            # starts at 1 and no component is in a gradient-zero saddle.
            init_vals = tfp.math.fill_triangular_inverse(
                tf.eye(n_mat, dtype=tf.float64)
            ).numpy()
            output_bias_init = tfk.initializers.Constant(init_vals)
        super().__init__(
            config,
            n_out,
            input_dim=input_dim,
            output_bias_init=output_bias_init,
            **kwargs,
        )


@tfk.utils.register_keras_serializable()
class SchwarzschildGlobalModel(BaseGlobalModel):
    """
    Single-submodel global metric network for the S^2-embedding architecture.

    The S^2 part of the input (q1, q2) is lifted to 3D Cartesian coordinates
    on the unit sphere via the hemisphere-aware inverse stereographic projection
    before being passed to the network.  The network receives a 5D input
    (T, X, X_cart, Y_cart, Z_cart) and outputs 15 Cholesky components of a
    5D ambient metric G_{AB} on R^2 x R^3.  The actual 4D intrinsic metric is
    obtained by pulling G back to the stereographic chart coordinates via the
    analytic embedding Jacobian: g_{mn} = G_{AB} J^A_m J^B_n.
    """

    def __init__(self, config: BaseConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        # Number of independent lower-triangular entries for the 5D AMBIENT metric
        # whose pullback gives the 4D intrinsic metric.  For dim=4: 5*6/2 = 15.
        n_out = int(0.5 * (self.dim + 1) * (self.dim + 2))

        # Single submodel with 5D Cartesian-embedded input
        self.submodel = SchwarzschildPatchSubModel(self.config, n_out, input_dim=5)

    def call(self, inputs):
        # inputs: (batch, 5) = [T, X, q1, q2, patch_idx_float]
        q_4d = inputs[:, :4]
        patch_idx = tf.cast(inputs[:, 4], tf.int32)
        x_5d = embed_S2_coords(q_4d, patch_idx)
        return self.submodel(x_5d)


class SchwarzschildSupervisedWrapper(tfk.Model):
    """
    Pullback wrapper for supervised pre-training of SchwarzschildGlobalModel.

    The base model outputs 15 Cholesky components of the 5D ambient metric G_{AB}.
    This wrapper computes the pullback g_{mn} = G_{AB} J^A_m J^B_n and returns
    the 4D intrinsic metric flattened to (batch, 16), enabling plain MSE training
    directly against 4D metric targets — no indefinite-Cholesky gymnastics needed.
    """

    def __init__(
        self, base_model: SchwarzschildGlobalModel, lorentzian: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.lorentzian = lorentzian

    def call(self, inputs):
        from geometry.schwarzschild import embedding_jacobian_stereo
        from helper_functions.helper_functions import cholesky_from_vec

        G_5d_vec = self.base_model(inputs)  # (batch, 15)
        # The +1 shift on L[0,0] (index 5) is applied inside SchwarzschildPatchSubModel.call
        # so G_5d_vec already has the correct Lorentzian initialisation baked in.
        q_4d = inputs[:, :4]
        patch_idx = tf.cast(inputs[:, 4], tf.int32)
        G_5d = cholesky_from_vec(G_5d_vec, lorentzian=self.lorentzian)  # (batch, 5, 5)
        J = embedding_jacobian_stereo(q_4d, patch_idx)  # (batch, 5, 4)
        g_4d = tf.einsum("sAB,sAm,sBn->smn", G_5d, J, J)  # (batch, 4, 4)
        return tf.reshape(g_4d, [-1, 16])  # (batch, 16)


@tfk.utils.register_keras_serializable()
class SchwarzschildLocal2DModel(BaseGlobalModel):
    """Direct 2D local metric model (no Schwarzschild embedding)."""

    def __init__(self, config: BaseConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)

        if self.dim != 2:
            raise ValueError(
                "SchwarzschildLocal2DModel requires geometry.dim=2. "
                f"Got dim={self.dim}."
            )

        # Number of independent entries for a 2x2 symmetric metric via Cholesky.
        n_out = int(0.5 * self.dim * (self.dim + 1))
        self.submodel = SchwarzschildPatchSubModel(self.config, n_out, input_dim=2)

    def call(self, inputs):
        # inputs: (batch, 2)
        return self.submodel(inputs)


@tfk.utils.register_keras_serializable()
class SchwarzschildNetwork(BaseNetwork, SerialisableModel):
    """
    Represents a class for the machine learning processes used in training the
    global metric function across the patches. This object contains the metric
    neural network models as an attribute subclass via GlobalModel, otherwise
    containing functionality for training, validating, saving, logging.
    """

    def __init__(
        self,
        config: BaseConfig | SchwarzschildConfig,
        restore_hps: bool = False,
    ) -> None:
        # Init BaseNetwork class
        super().__init__(config, restore_hps)

        # Build the model

        if not hasattr(self, "model"):
            self.model = SchwarzschildGlobalModel(self.config)

        # Define the loss
        self.loss = TotalSchwarzschildLoss(config=self.config)

    def evaluate_loss(
        self, x, training=True, return_constituents=False, val_print=True
    ):
        if hasattr(self.loss, "set_epoch") and not tf.inside_function():
            self.loss.set_epoch(self.current_epoch, self.total_epochs)
        # The loss recomputes the metric internally via a double-tape pass, so
        # there is no need for a separate model forward pass here.
        return self.loss.call(
            self.model,
            x,
            metric_pred=None,
            return_constituents=return_constituents,
            val_print=val_print,
        )


@tfk.utils.register_keras_serializable()
class SchwarzschildLocal2DNetwork(BaseNetwork, SerialisableModel):
    """Training harness for local 2D Schwarzschild-style Lorentzian runs."""

    def __init__(
        self,
        config: BaseConfig | SchwarzschildConfig,
        restore_hps: bool = False,
    ) -> None:
        super().__init__(config, restore_hps)

        if not hasattr(self, "model"):
            self.model = SchwarzschildLocal2DModel(self.config)

        self.loss = TotalSchwarzschildLocal2DLoss(config=self.config)
