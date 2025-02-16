import tensorflow as tf
tf.keras.backend.set_floatx("float64")
from geometry.geometry import compute_ricci_tensor
from helper_functions.helper_functions import cholesky_from_vec, RadiusWeighting
from geometry.geometry import PatchChange_Coordinates_Ball, PatchChange_Metric_Ball

class TotalLoss:
    """
    Represents a class for computing the total training loss, which has 
    contributions from solving the Einstein equation, from satisfying overlap 
    conditions of patches, and from finiteness of the metric components.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - num_dimensions (int): Number of dimensions of the manifold.
    - n_patches (int): Number of patches in the manifold definition (1 or 2).
    - overlap_upperwidth (float): distance from the radial midpoint to the edge 
      of the patch area of interest, beyond here the radial filter devalues point contributions.
    - print_losses (bool): Whether to print batch loss values with each call.
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - einstein_multiplier (float): The weighted contribution of the Einstein losses to the total loss.
    - overlap_multiplier (float): The weighted contribution of the overlap losses to the total loss.
    - finiteness_multiplier (float): The weighted contribution of the finiteness losses to the total loss.
    - einstein_losses (list): The Einstein loss values across the patches.
    - overlap_loss (flaot): The overlap loss values for the model.
    - filter_hyperparameters (list): The defining parameters for the finiteness filter function.
    - finite_losses (list): The Finiteness loss values across the patches.

    Methods:
    - __init__(self, hp, print_losses): 
      Initializes the TotalLoss class with the respective loss hyperparameters, 
      initialising the loss components and multipliers.

    - call(self, x_vars, return_constituents, val_print): 
      Computes the total loss, which is defined as a weighted sum of the loss
      components: Einstein, Overlap, Finiteness.
    """
    def __init__(self, hp, print_losses=False):
        self.hp = hp
        self.num_dimensions = self.hp["dim"]
        self.n_patches = self.hp["n_patches"]
        self.overlap_upperwidth = self.hp["overlap_upperwidth"]
        self.print_losses = print_losses

        # Einstein constant, $\lambda$ in the Einstein equation: $R_{ij} = \lambda g_{ij}$
        self.einstein_constant = self.hp["einstein_constant"]

        # Loss multipliers 
        self.einstein_multiplier = self.hp["einstein_multiplier"]
        self.overlap_multiplier = self.hp["overlap_multiplier"]
        self.finiteness_multiplier = self.hp["finiteness_multiplier"]
        assert abs(self.einstein_multiplier) + abs(self.overlap_multiplier) + abs(self.finiteness_multiplier) > 0., "All loss terms turned off..."
        if self.n_patches == 1:
            self.overlap_multiplier = tf.cast(0., tf.float64)

        # Einstein Loss
        if self.n_patches == 1:
            self.einstein_losses = [EinsteinLoss(self.num_dimensions, self.einstein_constant, False)]
        else:
            self.einstein_losses = [EinsteinLoss(self.num_dimensions, self.einstein_constant, True, self.overlap_upperwidth) for patch in range(int(self.n_patches))]
            
        # Overlap Loss
        if self.n_patches == 1:
            self.overlap_loss = tf.cast(0., tf.float64)
        elif self.n_patches == 2:
            self.overlap_loss = OverlapLoss(self.num_dimensions, True, self.overlap_upperwidth)
        else: 
            self.overlap_loss = tf.cast(0., tf.float64)
            print(f"Overlap loss not yet configured for {self.n_patches} patches...", flush=True)
            exit(1)
        
        # Finiteness Loss
        self.filter_hyperparameters = [
            self.hp["finite_centre"],
            self.hp["finite_width"],
            self.hp["finite_sharpness"],
            self.hp["finite_height"],
            self.hp["finite_slope"],
        ]
        self.finite_losses = [FiniteLoss(self.num_dimensions, self.filter_hyperparameters) for patch in range(int(self.n_patches))]


    def call(
        self, model, x_vars, metric_pred, return_constituents=False, val_print=True
    ):
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds = []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds.append(patch_1_output)
            metric_preds.append(patch_2_output)
        else:
            metric_preds.append(metric_pred)

        # Compute the loss components
        # Einstein
        if self.einstein_multiplier > 0.:
            e_losses = [self.einstein_losses[patch_idx].compute(patch_inputs[patch_idx], metric_preds[patch_idx], model.patch_submodels[patch_idx]) for patch_idx in range(int(self.n_patches))]
        else:
            e_losses = [tf.cast(0., tf.float64) for patch_idx in range(int(self.n_patches))]
           
        # Overlap
        if self.overlap_multiplier > 0. and self.n_patches == 2:
            overlap_loss = self.overlap_loss.compute(
                x_vars, [metric_preds[0], metric_preds[1]]
            )
        else:
            overlap_loss = tf.cast(0., tf.float64)
              
        # Finiteness
        if self.finiteness_multiplier > 0.:
            f_losses = [tf.math.log(self.finite_losses[patch_idx].compute(metric_preds[patch_idx])) for patch_idx in range(int(self.n_patches))]
        else:
            f_losses = [tf.cast(0., tf.float64) for patch_idx in range(int(self.n_patches))]
            
        # Print the batch loss values
        if self.print_losses and val_print:
            print(
                f"Einstein: {[tf.get_static_value(e_loss) for e_loss in e_losses]}\nOverlap: {tf.get_static_value(overlap_loss)}\nFinite: {[tf.get_static_value(f_loss) for f_loss in f_losses]}\n"
            )

        # Initialise the constituent losses dictionary (holds each of the
        # loss components pre-sum)
        if return_constituents:
            loss_constituents = {
                "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
                "overlap_loss": tf.get_static_value(overlap_loss),
                "finiteness_losses": [tf.get_static_value(f_loss) for f_loss in f_losses],
            }
        else:
            loss_constituents = None

        # Compute the total loss (accounting for multipliers)
        total_loss = 0.
        if self.einstein_multiplier > 0.:
            total_loss += self.einstein_multiplier * tf.reduce_sum(tf.math.abs(e_losses)) 
        if self.overlap_multiplier > 0.:
            total_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        if self.finiteness_multiplier > 0.:
            total_loss += self.finiteness_multiplier * tf.reduce_sum(tf.math.abs(f_losses)) 
        # Normalise by the multiplier factors
        total_loss /= self.einstein_multiplier + self.overlap_multiplier + self.finiteness_multiplier


        return total_loss, loss_constituents


class EinsteinLoss:
    """
    Represents a class for computing the Einstein loss, which measures the
    difference between the Ricci tensor and the predicted metric tensor
    (scaled by the Einstein constant $\lambda$).

    Attributes:
    - dim (int): The dimensionality of the metric tensor (number of x coords).
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - weight_radially (bool): Whether to apply radial weighting to the points.
    - overlap_upperwidth (float): distance from the radial midpoint to the edge 
      of the patch area of interest, beyond here the radial filter devalues point contributions.

    Methods:
    - __init__(self, num_dimensions, einstein_constant, weight_radially, overlap_upperwidth): 
      Initializes the EinsteinLoss class with the respective loss hyperparameters.

    - compute(self, x_vars, metric_pred, model): 
      Computes the Einstein loss, which is defined as the squared norm of the 
      difference between the Ricci tensor and the predicted metric tensor 
      (scaled by a constant).
    """
    def __init__(self, num_dimensions, einstein_constant=1.0, weight_radially=True, overlap_upperwidth=0.1) -> None:
        self.dim = num_dimensions
        self.einstein_constant = einstein_constant
        self.weight_radially = weight_radially
        self.overlap_upperwidth = overlap_upperwidth

    def compute(self, x_vars, metric_pred, model):
        # Compute the Ricci tensor
        ricci_tensor = compute_ricci_tensor(x_vars, model)
        # Convert the metric vielbein to a matrix
        metric_pred_mat = cholesky_from_vec(metric_pred)
        # Compute the loss from the Einstein equation
        norm = tf.norm(
            self.einstein_constant * metric_pred_mat - ricci_tensor, axis=(1, 2)
        )
        
        # Apply radial weighting
        if self.weight_radially:
            radial_midpoint = tf.cast(tf.sqrt(3. - 2 * tf.sqrt(2.)), tf.float64)
            filter_width = radial_midpoint + self.overlap_upperwidth
            radial_weights = RadiusWeighting(x_vars, filter_width)
            norm = norm * radial_weights
        
        einstein_loss = tf.reduce_mean(norm)

        return einstein_loss


class OverlapLoss:
    """
    Represents a class for computing the overlap loss, which measures the
    difference between agreement of the metric predictions between the patches.
    This uses symmetric contributions from the difference between the metric 
    prediction in patch 1 and the metric prediction for patch 2 transformed 
    into patch 1, and equivalently the difference between the metric prediction 
    in patch 2 and the metric prediction for patch 1 transformed into patch 2.
    The contributions are weighted by the points radial positions, prioirtising 
    points within the overlap region which is an annulus about the radial midpoint.

    Attributes:
    - dim (int): The dimensionality of the metric tensor (number of x coords).
    - weight_radially (bool): Whether to apply radial weighting to the points.
    - overlap_upperwidth (float): distance from the radial midpoint to the edge 
      of the patch area of interest, outisde of here and the transformed lower 
      bound, the radial filter devalues point contributions.

    Methods:
    - __init__(self, num_dimensions, weight_radially, overlap_upperwidth): 
      Initializes the OverlapLoss class with the respective loss hyperparameters.

    - compute(self, x_vars, metric_pred): 
      Computes the overlap loss, which is defined as the norm of the 
      differences between metric preditions in each patch and their transforms 
      between these patches, weighted to prioiritise points in the overlap region.
    """
    def __init__(self, num_dimensions, weight_radially=True, overlap_upperwidth=0.1) -> None:
        self.dim = num_dimensions
        self.weight_radially = weight_radially
        self.overlap_upperwidth = overlap_upperwidth

    def compute(self, x_vals, metric_preds):
        # Convert the outputs to metrics
        patch_1_metric_pred = tf.map_fn(cholesky_from_vec, metric_preds[0])
        patch_2_metric_pred = tf.map_fn(cholesky_from_vec, metric_preds[1])

        # Compute the patch changes of the both outputs
        patch_2_metrics_from_patch_1 = PatchChange_Metric_Ball(
            x_vals, patch_1_metric_pred
        )
        patch_1_metrics_from_patch_2 = PatchChange_Metric_Ball(
            PatchChange_Coordinates_Ball(x_vals), patch_2_metric_pred
        )

        # Take the total difference in both patches between the metrics in both patches
        overlap_loss = tf.reduce_mean(
            abs(patch_2_metrics_from_patch_1 - patch_2_metric_pred), axis=(1,2)
        ) + tf.reduce_mean(
            abs(patch_1_metrics_from_patch_2 - patch_1_metric_pred), axis=(1,2)
        )
            
        # Apply radial weighting
        if self.weight_radially:
            radial_midpoint = tf.cast(tf.sqrt(3. - 2 * tf.sqrt(2.)), tf.float64)
            filter_lower_bound = (1 - (radial_midpoint + self.overlap_upperwidth)) / (1 + (radial_midpoint + self.overlap_upperwidth))
            filter_midpoint = ((radial_midpoint + self.overlap_upperwidth) + filter_lower_bound) / 2.
            filter_width = radial_midpoint + self.overlap_upperwidth - filter_midpoint
            radial_weights = RadiusWeighting(x_vals, filter_width=filter_width, filter_midpt=filter_midpoint)
            overlap_loss = radial_weights * overlap_loss
            
        overlap_loss = tf.reduce_mean(overlap_loss)

        return overlap_loss
    
    
class FiniteLoss:
    """
    Represents a class for computing the finiteness loss, which measures the
    norm of the metric components and weights according to a predefined filter.
    This loss component ensures the zero metric is avoided as an attractor point
    of the learning.

    Attributes:
    - dim (int): The dimensionality of the metric tensor (number of x coords).
    - filter_hyperparameters (list): The defining parameters for the finiteness filter function.

    Methods:
    - __init__(self, num_dimensions, filter_hyperparameters): 
      Initializes the FinitenessLoss class with the respective loss hyperparameters.

    - compute(self, metric_pred): 
      Computes the Finiteness loss, which is defined as the norm of the 
      metric components after applying a filter weighting function.
    """
    def __init__(self, num_dimensions, filter_hyperparameters) -> None:
        self.dim = num_dimensions
        self.filter_hyperparameters = filter_hyperparameters

    def compute(self, metric_pred):
        # Convert the metric vielbein to a matrix
        metric_pred_mat = cholesky_from_vec(metric_pred)

        # Import filter hyperparameters
        finite_centre = self.filter_hyperparameters[0]
        finite_width = self.filter_hyperparameters[1]
        finite_sharpness = self.filter_hyperparameters[2]
        finite_height = self.filter_hyperparameters[3]
        finite_slope = self.filter_hyperparameters[4]

        # Compute the norm of the metric components
        sum_metric_pred = tf.reduce_sum(
            abs(metric_pred_mat), axis=[1, 2], keepdims=True
        ) * 2 / ((self.dim)*(self.dim -1))

        # Define the finiteness filter weighting function
        gaussian_weight = (
            (
                tf.square(
                    finite_height
                    * tf.exp(
                        -tf.pow(
                            ((sum_metric_pred - finite_centre) / finite_width),
                            finite_sharpness,
                        )
                    )
                    - finite_height
                )
                + 1
            )
            + (
                sum_metric_pred / finite_slope
                - (finite_centre + finite_width) / finite_slope
            )
            * (
                1
                + tf.math.tanh(sum_metric_pred / 2 - (finite_centre + finite_width) / 2)
            )
            / 2
            + (
                -sum_metric_pred / finite_slope
                + (finite_centre - finite_width) / finite_slope
            )
            * (
                1
                + tf.math.tanh(
                    -sum_metric_pred / 2 + (finite_centre - finite_width) / 2
                )
            )
            / 2
        )

        finite_loss = tf.square(1 - tf.reduce_mean(gaussian_weight)) + 1

        return finite_loss
    

class GlobalLoss:
    """
    Represents a class for computing the global test loss, which has 
    contributions from solving the Einstein equation and from satisfying overlap 
    conditions of patches. The patches are restricted to points within the radial 
    limit, and the overlap region is an annulus which spans either side of the 
    radial midpoint and runs up to the radial limit, such that it is symmetric 
    under the patch transform function.

    Attributes:
    - hp (dict): Dictionary of the training hyperparameters.
    - num_dimensions (int): Number of dimensions of the manifold.
    - n_patches (int): Number of patches in the manifold definition (1 or 2).
    - radial_limit (float): The hard radial boundary for the patches.
    - radial_midpoint (float): The radial value which maps to itself under the function which transofrms between patches.
    - einstein_constant (float): The proportioanlity constant in the Einstein equation.
    - einstein_multiplier (float): The weighted contribution of the Einstein losses to the total loss.
    - overlap_multiplier (float): The weighted contribution of the overlap losses to the total loss.
    - einstein_losses (list): The Einstein loss values across the patches.
    - overlap_loss (float): The overlap loss values for the model.

    Methods:
    - __init__(self, hp, radial_limit): 
      Initializes the GlobalLoss class with the respective loss hyperparameters, 
      initialising the loss components and multipliers.

    - call(self, model, x_vars, metric_pred): 
      Computes the global test loss, which is defined as a weighted sum of the loss
      components: Einstein, Overlap. 
    """
    def __init__(self, hp, radial_limit=None):
        self.hp = hp
        self.num_dimensions = self.hp["dim"]
        self.n_patches = self.hp["n_patches"]
        self.radial_limit = radial_limit
        self.radial_midpoint = tf.cast(tf.sqrt(3. - 2 * tf.sqrt(2.)), tf.float64)
        # Ensure the patching conditions are consistently defined
        if self.radial_limit:
            assert self.radial_limit > self.radial_midpoint, "Patches do not overlap..."

        # Einstein constant, $\lambda$ in the Einstein equation: $R_{ij} = \lambda g_{ij}$
        self.einstein_constant = self.hp["einstein_constant"]

        # Loss multipliers
        self.einstein_multiplier = self.hp["einstein_multiplier"]
        self.overlap_multiplier = self.hp["overlap_multiplier"]
        if self.n_patches == 1:
            self.overlap_multiplier = tf.cast(0., tf.float64)

        # Einstein Loss
        self.einstein_losses = [EinsteinLoss(self.num_dimensions, self.einstein_constant, weight_radially=False) for i in range(int(self.n_patches))]

        # Overlap Loss
        if self.n_patches == 1:
            self.overlap_loss = tf.cast(0., tf.float64)
        elif self.n_patches == 2:
            self.overlap_loss = OverlapLoss(self.num_dimensions, weight_radially=False)
        else: 
            self.overlap_loss = 0.
            print(f"Overlap loss not yet configured for {self.n_patches} patches...", flush=True)

    def call(
        self, model, x_vars, metric_pred
    ):     
        # Set up the network inputs & outputs
        patch_inputs = [x_vars]
        metric_preds = []
        if self.n_patches > 1:
            # Compute the input coordinates in the second patch
            patch_inputs.append(model.patch_transform_layer(x_vars))

            # Split the output into the metrics in each patch
            patch_1_output, patch_2_output = tf.split(
                metric_pred, num_or_size_splits=2, axis=-1
            )
            metric_preds.append(patch_1_output)
            metric_preds.append(patch_2_output)
        else:
            metric_preds.append(metric_pred)
        
        #Compute data limited to each patch
        if self.radial_limit and self.radial_limit > 0:
            # Patches
            norms = [tf.sqrt(tf.reduce_sum(tf.square(p_pts), axis=1)) for p_pts in patch_inputs]
            masks = [norm < self.radial_limit for norm in norms] #...find points within the radial limit
            pts_limited = [tf.boolean_mask(patch_inputs[p_idx], masks[p_idx]) for p_idx in range(int(self.n_patches))]
            metrics_limited = [tf.boolean_mask(metric_preds[p_idx], masks[p_idx]) for p_idx in range(int(self.n_patches))]

            # Overlap Region
            mask_overlap = tf.logical_and(norms[0] >= (1 - self.radial_limit) / (1 + self.radial_limit), 
                                          norms[0] <= self.radial_limit) #...find points within the overlap region
            pts_overlap = tf.boolean_mask(patch_inputs[0], mask_overlap)
            metrics_overlap = [tf.boolean_mask(metric_preds[p_idx], mask_overlap) for p_idx in range(int(self.n_patches))]
        else:
            #...otherwise use the full patches in each case
            pts_limited, metrics_limited, pts_overlap, metrics_overlap = patch_inputs, metric_preds, patch_inputs[0], metric_preds
                        
        # Compute the number of points in each region
        sample_sizes = [[p_pts.shape[0] for p_pts in pts_limited], pts_overlap.shape[0]]

        # Compute the loss components        
        if self.einstein_multiplier > 0.:
            e_losses = [self.einstein_losses[patch_idx].compute(pts_limited[patch_idx], metrics_limited[patch_idx], model.patch_submodels[patch_idx]) for patch_idx in range(int(self.n_patches))]
        else:
            e_losses = [tf.cast(0., tf.float64) for patch_idx in range(int(self.n_patches))]
        if self.overlap_multiplier > 0. and self.n_patches > 1:
            overlap_loss = self.overlap_loss.compute(pts_overlap, metrics_overlap)
        else:
            overlap_loss = tf.cast(0., tf.float64)

        # Return loss components
        loss_constituents = {
                "einstein_losses": [tf.get_static_value(e_loss) for e_loss in e_losses],
                "overlap_loss": tf.get_static_value(overlap_loss)
            }

        # Compute the total loss (accounting for multiplier)
        global_loss = 0.
        if self.einstein_multiplier > 0.:
            global_loss += self.einstein_multiplier * tf.reduce_sum(tf.math.abs(e_losses)) 
        if self.overlap_multiplier > 0.:
            global_loss += self.overlap_multiplier * tf.math.abs(overlap_loss)
        global_loss /= self.einstein_multiplier + self.overlap_multiplier

        return global_loss, loss_constituents, sample_sizes
    
    
    
    