import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

class KargerExchange:
    def __init__(self, models):
        """
        Kärger Exchange model coupling multiple compartments.

        Parameters
        ----------
        models : list
            List of instantiated JAX model objects (e.g. [C1Stick(), G1Ball()]).
            Each model must have `predict` or `__call__` and `parameter_names`.
        """
        self.models = models
        self.n_models = len(models)
        
        # Construct parameter handling
        self.parameter_names = []
        self.parameter_cardinality = {}
        self.parameter_ranges = {}
        
        # 1. Sub-model parameters
        for i, model in enumerate(models):
            for name in model.parameter_names:
                # We prefix parameter names to avoid collisions if identical models are used
                # e.g. "model0_diffusivity"
                new_name = f"model{i}_{name}"
                self.parameter_names.append(new_name)
                self.parameter_cardinality[new_name] = model.parameter_cardinality[name]
                if hasattr(model, 'parameter_ranges'):
                     self.parameter_ranges[new_name] = model.parameter_ranges[name]
        
        # 2. Fractions (N-1)
        # We generally expect N fractions summing to 1. 
        # For optimization, we usually parameterize N-1.
        # But per the prompt Requirements: "partial volume fractions f".
        # We'll explicitly ask for N-1 fraction parameters: `partial_volume_0`, `partial_volume_1`...
        # The last one is implicit.
        for i in range(self.n_models - 1):
            name = f"partial_volume_{i}"
            self.parameter_names.append(name)
            self.parameter_cardinality[name] = 1
            self.parameter_ranges[name] = (0.0, 1.0)
            
        # 3. Exchange times
        # For N=2, we need 1 exchange time (tau_12).
        # For N=3, we need 3 exchange times (tau_12, tau_13, tau_23).
        # General formula: N * (N - 1) / 2
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                name = f"exchange_time_{i}{j}"
                self.parameter_names.append(name)
                self.parameter_cardinality[name] = 1
                self.parameter_ranges[name] = (0.0, jnp.inf) # TODO: sensible upper bound?

    def _unpack_params(self, params):
        """
        Unpacks flat params array into dictionary of model params, fractions, and exchange times.
        """
        param_dict = {}
        idx = 0
        
        # 1. Sub-model params
        submodel_params_list = []
        for i, model in enumerate(self.models):
            model_p = {}
            for name in model.parameter_names:
                card = model.parameter_cardinality[name]
                if card == 1:
                    val = params[idx]
                    idx += 1
                else:
                    val = params[idx : idx + card]
                    idx += card
                model_p[name] = val
            submodel_params_list.append(model_p)
            
        # 2. Fractions
        fractions = []
        for i in range(self.n_models - 1):
            fractions.append(params[idx])
            idx += 1
        
        # Calculate last fraction
        fractions = jnp.array(fractions)
        last_frac = 1.0 - jnp.sum(fractions)
        # Ensure we clamp or handle numerical issues? 
        # Ideally the optimizer respects bounds. 
        # For safety inside the function we can't easily clamp without potentially breaking gradients 
        # if the sum > 1. 
        # We will assume valid inputs.
        all_fractions = jnp.append(fractions, last_frac)
        
        # 3. Exchange times
        exchange_times = {} # Keyed by (i, j) tuple
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                val = params[idx]
                idx += 1
                exchange_times[(i, j)] = val
                
        return submodel_params_list, all_fractions, exchange_times

    def prediction_function(self, params, acquisition):
        """
        Pure JAX function implementing the Karger Matrix Exponential.
        This is the inner function to be vmapped.
        
        Parameters
        ----------
        params : 1D array
            Flat parameters.
        bval : float
            Single b-value.
        grad : 1D array
            Single gradient vector (3,).
        delta : float
            Diffusion time (Delta).
            
        Wait, expm needs to run per voxel/acquisition point.
        But `jax.scipy.linalg.expm` runs on a single matrix.
        So we need to construct the matrix for one acquisition point.
        """
        # The logic below splits execution:
        # 1. Unpack params (global for the voxel).
        # 2. Calculate uncoupled signals for 'acquisition'.
        # 3. Form matrices and solve.
        pass

    def __call__(self, bvals, gradient_directions, diffusion_time, **kwargs):
        """
        Main entry point.
        Supports inputs as arrays (voxels * M measurements).
        JAX models typically take parameters as explicit arguments in kwargs matching `parameter_names`.
        
        However, the `predict` requirement in the prompt implies a specific signature:
        `predict(acquisition_params, model_params)`
        
        Standard dmipy_jax models seem to follow `__call__(bvals, vectors, **params)`.
        
        BUT, this Generic Exchange model is best implemented as a function that takes 
        a flat parameter vector (for optimization) OR loose arguments.
        
        Given the prompt asks for "predict", I will implement `predict`.
        """
        pass
    
    def predict(self, params, acquisition):
        """
        Predicts the signal attenuation using the Karger Matrix Exponential involved solution.
        
        Parameters
        ----------
        params : jax.numpy.ndarray
            1D array of parameters for a single voxel.
        acquisition : JaxAcquisition
            Acquisition scheme parameters.
            
        Returns
        -------
        signal : jax.numpy.ndarray
            1D array of signal attenuation of shape (N_measurements,).
        """
        submodel_params, fractions, exchange_times = self._unpack_params(params)
        
        # 1. Compute Uncoupled Sub-model Signals
        # For each model, calculating S_i for all measurements
        uncoupled_signals = []
        for i, model in enumerate(self.models):
            # We assume models can take standard kwargs. 
            # Note: We pass only relevant sub-model params unpacked earlier.
            s_i = model(
                bvals=acquisition.bvalues,
                gradient_directions=acquisition.gradient_directions,
                delta=acquisition.delta,
                Delta=acquisition.Delta,
                small_delta=acquisition.delta,
                big_delta=acquisition.Delta,
                **submodel_params[i]
            )
            uncoupled_signals.append(s_i)
            
        # Stack -> (N_models, N_meas)
        uncoupled_signals = jnp.stack(uncoupled_signals, axis=0) 
        
        # Clip to avoid log(0)
        uncoupled_signals = jnp.clip(uncoupled_signals, 1e-20, 1.0)
        
        # R_diagonals (N_meas, N_models) = ln(S_i)
        # Note: This term represents -b * D_app (attenuation exponent).
        R_diagonals = jnp.log(uncoupled_signals).T 
        
        # 2. Construct Exchange Matrix K (N_models, N_models)
        K = self._construct_exchange_matrix(fractions, exchange_times)
        
        # 3. Assemble System Matrix Lambda
        # Lambda = diag(R) + K * Delta
        # K has units 1/s. R is unitless. Delta has units s.
        # We need diffusion time.
        if acquisition.Delta is None:
            # Fallback for synthetic cases without Delta?
            # We assume Delta=1.0 if not provided, implies K is in units of 1/acquisition_time?
            # Or raise error.
            # Given functional nature, let's substitute 1.0 but warn? 
            # Dmipy standardly requires Delta for exchange.
            Delta = 1.0
        else:
            Delta = acquisition.Delta
            
        # Delta might be shape (N_meas,) or scalar
        if jnp.ndim(Delta) == 0:
            Delta = Delta * jnp.ones_like(acquisition.bvalues) # (N_meas,)
            
        # K_eff: (N_meas, N_models, N_models)
        # We want K * Delta[m].
        # K is (N, N).
        K_eff = K[None, :, :] * Delta[:, None, None]
        
        # Lambda: (N_meas, N_models, N_models)
        # diag(R): (N_meas, N, N)
        Lambda = jax.vmap(jnp.diag)(R_diagonals) + K_eff
        
        
        # 4. Solve System
        # S = 1^T . expm(Lambda) . M0
        # M0 = fractions (N_models,)
        M0 = fractions
        
        # Compute expm for each measurement matrix
        # Explicit VMAP as per requirement to ensure batching works correctly
        E = jax.vmap(expm)(Lambda) # (N_meas, N_models, N_models)
        
        # E . M0 -> (N_meas, N_models)
        # M0 is (N,) broadcasted to (N_meas, N)
        # Result (M, N). [E_ij M0_j]
        evolved_magnetization = jnp.einsum('mnk,k->mn', E, M0)
        
        # Sum over compartments -> (N_meas,)
        signal = jnp.sum(evolved_magnetization, axis=1)
        
        return signal

    def _construct_exchange_matrix(self, f, exchange_times):
        """
        Constructs K (N, N) from fractions and exchange parameters.
        Enforces detailed balance: f_i K_ij = f_j K_ji.
        Constraints: row sums = 0.
        
        Parametrization (N=2):
        tau_12 = residence time of 1 exchanging to 2.
        K_12 = 1/tau_12.
        K_21 = K_12 * (f_1 / f_2).
        K_11 = -K_12.
        K_22 = -K_21.
        
        Parametrization (Generic):
        Input `exchange_times` {(i,j): tau_ij} for i<j.
        K_ij = 1 / tau_ij   (rate i -> j)
        K_ji = K_ij * (f_i / f_j) (rate j -> i, enforced by balance)
        
        Diagonal K_ii = - sum_{j!=i} K_ij
        """
        N = self.n_models
        K = jnp.zeros((N, N))

        # dM/dt = K M with M a column vector, so K[j, i] is the rate i -> j and
        # K[i, i] = -sum_j rate(i -> j). Column sums are then zero and total
        # magnetisation is conserved (E(b=0) = 1). Detailed balance fixes the
        # reverse rate: f_i k_ij = f_j k_ji.
        #
        # An earlier version also wrote the off-diagonals with ``.set`` before
        # this loop added them again, doubling the transfer terms relative to
        # the diagonals; the resulting matrix pumped magnetisation (1.49 at b=0).
        rows = []
        cols = []
        vals = []
        
        for (i, j), tau in exchange_times.items():
            rate_i_j = 1.0 / (tau + 1e-9)
            rate_j_i = rate_i_j * (f[i]/(f[j] + 1e-12)) # Safeguard
            
            # i -> j: K[j, i] += rate, K[i, i] -= rate
            rows.extend([j, i])
            cols.extend([i, i])
            vals.extend([rate_i_j, -rate_i_j])
            
            # j -> i: K[i, j] += rate, K[j, j] -= rate
            rows.extend([i, j])
            cols.extend([j, j])
            vals.extend([rate_j_i, -rate_j_i])
            
        if rows:
            rows = jnp.array(rows)
            cols = jnp.array(cols)
            vals = jnp.array(vals)
            K = K.at[rows, cols].add(vals)
            
        return K

