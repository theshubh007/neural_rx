# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

##### Neural Receiver #####

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, Layer
# from tensorflow.nn import relu
from sionna.utils import (
    flatten_last_dims,
)
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import (
    TBDecoder,
    LayerDemapper,
)
import tensorflow as tf


def to_numpy(input_array):
    # Check if the input is already a NumPy array
    if isinstance(input_array, np.ndarray):
        return input_array

    # Check if the input is a TensorFlow tensor
    try:
        import tensorflow as tf

        if isinstance(input_array, tf.Tensor):
            return input_array.numpy()
    except ImportError:
        pass

    # Check if the input is a PyTorch tensor
    try:
        import torch

        if isinstance(input_array, torch.Tensor):
            return input_array.cpu().numpy()
    except ImportError:
        pass

    raise TypeError(
        "Input type not supported. Please provide a NumPy array, TensorFlow tensor, or PyTorch tensor."
    )


class StateInit(nn.Module):
    r"""
    Network initializing the state tensor for each user.

    The network consists of len(num_units) hidden blocks, each block
    consisting of
    - A Conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity.
    """

    def __init__(
        self,
        d_s,
        num_units,
        in_channels,
        layer_type="sepconv",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()

        # Allows for the configuration of multiple layer types
        if layer_type == "sepconv":
            layer = (
                nn.Conv2d
            )  # PyTorch doesn't have SeparableConv2D natively; using Conv2d
        elif layer_type == "conv":
            layer = nn.Conv2d
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = nn.ModuleList()
        for n in num_units:
            # Using padding=1 to simulate 'same' padding with 3x3 kernel
            conv = layer(in_channels, n, kernel_size=(3, 3), padding=1)
            self._hidden_conv.append(conv)
            in_channels = n  # Update in_channels for the next layer

        # Output block
        self._output_conv = layer(in_channels, d_s, kernel_size=(3, 3), padding=1)

    def forward(self, inputs):
        y, pe, h_hat = inputs
        batch_size = y.shape[0]
        num_tx = pe.shape[0]

        # Stack the inputs
        y = y.unsqueeze(1).repeat(1, num_tx, 1, 1, 1)
        y = y.view(-1, *y.shape[2:])

        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        pe = pe.view(-1, *pe.shape[2:])

        # Ignore h_hat if no channel estimate is provided
        if h_hat is not None:
            h_hat = h_hat.view(-1, *h_hat.shape[2:])
            z = torch.cat([y, pe, h_hat], dim=-1)
        else:
            z = torch.cat([y, pe], dim=-1)

        # Apply the neural network
        for conv in self._hidden_conv:
            z = F.relu(conv(z))
        z = self._output_conv(z)

        # Unflatten
        s0 = z.view(batch_size, num_tx, *z.shape[1:])
        return s0  # Initial state of every user


class AggregateUserStates(nn.Module):
    r"""
    For every user n, aggregate the states of all the other users n' != n.

    An MLP is applied to every state before aggregating.
    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `d_s` units.
    """

    def __init__(
        self,
        d_s,
        num_units,
        in_features,
        layer_type="dense",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Ensure in_features is passed as an integer
        if not isinstance(in_features, int):
            print(f"in_features: {in_features}")
            raise TypeError(f"in_features must be an int but got {type(in_features)}")

        # Initialize hidden layers with both in_features and out_features
        self._hidden_layers = nn.ModuleList()
        for n in num_units:
            # Check if both in_features and n are integers
            if not isinstance(n, int):
                raise TypeError(f"out_features (n) must be an int but got {type(n)}")
            self._hidden_layers.append(layer(in_features, n))
            in_features = n  # Update in_features for the next layer

        # Output layer
        self._output_layer = layer(in_features, d_s)

    def forward(self, inputs):
        print("AggregateUserStates forward")
        s, active_tx = inputs

        # Process s
        sp = s
        print("flag1")
        for layer in self._hidden_layers:
            sp = F.relu(layer(sp))
        sp = self._output_layer(sp)
        print("flag2")

        # Mask non-active users
        active_tx = active_tx.unsqueeze(-1).expand_as(sp)
        sp = sp * active_tx

        # Aggregate and remove self-state
        a = sp.sum(dim=1, keepdim=True) - sp

        # Scale by the number of active users
        p = active_tx.sum(dim=1, keepdim=True) - 1.0
        p = F.relu(p)  # Clip negative values to ignore non-active users

        # Avoid division by zero
        p = torch.where(p == 0.0, torch.tensor(1.0), 1.0 / p)
        a = a * p
        print("flag3")

        return a


class UpdateState(nn.Module):
    r"""
    Updates the state tensor.

    The network consists of len(num_units) hidden blocks, each block i
    consisting of
    - A Conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity.

    The network ends with a skip connection with the state.
    """

    def __init__(
        self, d_s, num_units, layer_type="sepconv", dtype=torch.float32, **kwargs
    ):
        super().__init__()

        if layer_type == "sepconv":
            layer = nn.Conv2d  # Placeholder for SeparableConv2d
        elif layer_type == "conv":
            layer = nn.Conv2d
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = nn.ModuleList(
            [
                layer(n, kernel_size=(3, 3), padding="same", dtype=dtype)
                for n in num_units
            ]
        )

        # Output block
        self._output_conv = layer(d_s, kernel_size=(3, 3), padding="same", dtype=dtype)

    def forward(self, inputs):
        s, a, pe = inputs
        batch_size = s.shape[0]
        num_tx = s.shape[1]

        # Stack the inputs
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        pe = pe.view(-1, *pe.shape[2:])
        s = s.view(-1, *s.shape[2:])
        a = a.view(-1, *a.shape[2:])
        z = torch.cat([a, s, pe], dim=-1)

        # Apply the neural network
        for conv in self._hidden_conv:
            z = F.relu(conv(z))
        z = self._output_conv(z)

        # Skip connection
        z = z + s

        # Unflatten
        s_new = z.view(batch_size, num_tx, *z.shape[1:])
        return s_new  # Updated tensor state for each user


class CGNNIt(nn.Module):
    def __init__(
        self,
        d_s,
        num_units_agg,
        num_units_state_update,
        layer_type_dense="dense",
        layer_type_conv="sepconv",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        self._state_aggreg = None
        self._d_s = d_s
        self._num_units_agg = num_units_agg
        self._layer_type_dense = layer_type_dense
        self._dtype = dtype

    def forward(self, inputs):
        # Pass a dummy input through to calculate in_features dynamically
        s, active_tx = inputs
        in_features = s.shape[-1]  # Assuming last dimension is the input size

        if self._state_aggreg is None:
            # Initialize state_aggreg with inferred in_features
            self._state_aggreg = AggregateUserStates(
                self._d_s,
                self._num_units_agg,
                in_features,
                self._layer_type_dense,
                dtype=self._dtype,
            )
        # Forward pass through state_aggreg
        return self._state_aggreg((s, active_tx))


class ReadoutLLRs(nn.Module):
    r"""
    Network computing LLRs from the state vectors.

    This is an MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.
    """

    def __init__(
        self,
        num_bits_per_symbol,
        num_units,
        in_features,  # You need to pass the input feature size
        layer_type="dense",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()

        # Choose the correct layer type
        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Initialize the hidden layers
        self._hidden_layers = nn.ModuleList()

        # Set up the first layer with the given input feature size
        for n in num_units:
            self._hidden_layers.append(layer(in_features, n))
            in_features = n  # Update in_features for the next layer

        # Output layer
        self._output_layer = layer(in_features, num_bits_per_symbol)

    def forward(self, s):
        # Apply hidden layers
        for layer in self._hidden_layers:
            s = F.relu(layer(s))

        # Apply output layer
        llr = self._output_layer(s)

        return llr


class ReadoutChEst(nn.Module):
    r"""
    Network computing channel estimate.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `2*num_rx_ant` units.
    """

    def __init__(
        self,
        num_rx_ant,
        num_units,
        in_features,
        layer_type="dense",
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden layers
        self._hidden_layers = nn.ModuleList()
        for n in num_units:
            self._hidden_layers.append(
                layer(in_features, n)
            )  # Pass in_features and out_features
            in_features = n  # Update in_features for the next layer

        # Output layer
        self._output_layer = layer(
            in_features, 2 * num_rx_ant
        )  # Output size is `2 * num_rx_ant`

    def forward(self, s):
        # Apply the MLP layers
        z = s
        for layer in self._hidden_layers:
            z = F.relu(layer(z))
        h_hat = self._output_layer(z)

        return h_hat


class CGNN(nn.Module):
    r"""
    Implements the core neural receiver consisting of
    convolutional and graph layer components (CGNN).
    """

    def __init__(
        self,
        num_bits_per_symbol,
        num_rx_ant,
        num_it,
        d_s,
        num_units_init,
        num_units_agg,
        num_units_state,
        num_units_readout,
        layer_type_dense,
        layer_type_conv,
        layer_type_readout,
        training=False,
        apply_multiloss=False,
        var_mcs_masking=False,
        dtype=torch.float32,
        **kwargs,
    ):
        super().__init__()
        # Readouts
        in_features = d_s  # `d_s` should be the input feature size to the readout
        self._training = training
        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking

        # Define in_channels - this should match the number of channels in the input data
        in_channels = (
            16  # Replace 16 with the actual number of input channels in your data
        )

        # Initialization for the state
        if self._var_mcs_masking:
            self._s_init = [
                StateInit(
                    d_s,
                    num_units_init,
                    in_channels,
                    layer_type=layer_type_conv,
                    dtype=dtype,
                )
            ]
        else:
            self._s_init = (
                nn.ModuleList()
            )  # Use ModuleList for multiple initializations
            for _ in num_bits_per_symbol:
                self._s_init.append(
                    StateInit(
                        d_s,
                        num_units_init,
                        in_channels,
                        layer_type=layer_type_conv,
                        dtype=dtype,
                    )
                )

        # Iteration blocks
        self._iterations = nn.ModuleList()
        for i in range(num_it):
            it = CGNNIt(
                d_s,
                num_units_agg[i],
                num_units_state[i],
                layer_type_dense=layer_type_dense,
                layer_type_conv=layer_type_conv,
                dtype=dtype,
            )
            self._iterations.append(it)
        self._num_it = num_it

        # Readouts
        in_features = d_s  # `d_s` should be the input feature size to the readout

        if self._var_mcs_masking:
            self._readout_llrs = [
                ReadoutLLRs(
                    max(num_bits_per_symbol),
                    num_units_readout,
                    in_features,  # Pass `in_features`
                    layer_type=layer_type_readout,
                    dtype=dtype,
                )
            ]
        else:
            self._readout_llrs = nn.ModuleList()
            for num_bits in num_bits_per_symbol:
                self._readout_llrs.append(
                    ReadoutLLRs(
                        num_bits,
                        num_units_readout,
                        in_features,  # Pass `in_features`
                        layer_type=layer_type_readout,
                        dtype=dtype,
                    )
                )

        # Channel estimate readout
        self._readout_chest = ReadoutChEst(
            num_rx_ant,
            num_units_readout,
            in_features=in_features,
            layer_type=layer_type_readout,
            dtype=dtype,
        )

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

    @property
    def apply_multiloss(self):
        """Average loss over all iterations or eval just the last iteration."""
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (
            val <= len(self._iterations)
        ), "Invalid number of iterations"
        self._num_it = val

    def forward(self, inputs):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs

        ########################################
        # Normalization
        #########################################
        # Normalize the input such that each batch sample has unit power
        norm_scaling = torch.mean(y**2, dim=(1, 2, 3), keepdim=True)
        norm_scaling = torch.rsqrt(norm_scaling)
        y = y * norm_scaling

        norm_scaling = norm_scaling.unsqueeze(1)
        if h_hat is not None:
            h_hat = h_hat * norm_scaling

        ########################################
        # State initialization
        ########################################
        if self._var_mcs_masking:
            s = self._s_init[0]((y, pe, h_hat))
        else:
            s = self._s_init[0]((y, pe, h_hat)) * mcs_ue_mask[:, :, 0:1].unsqueeze(-1)
            for idx in range(1, self._num_mcss_supported):
                s = s + self._s_init[idx]((y, pe, h_hat)) * mcs_ue_mask[
                    :, :, idx : idx + 1
                ].unsqueeze(-1)

        ########################################
        # Run receiver iterations
        ########################################
        llrs = []
        h_hats = []
        for i in range(self._num_it):
            it = self._iterations[i]
            # State update
            s = it([s, pe, active_tx])

            # Read-outs
            if (self._training and self._apply_multiloss) or i == self._num_it - 1:
                llrs_ = []
                # Iterate over all MCS schemes individually
                for idx in range(self._num_mcss_supported):
                    if self._var_mcs_masking:
                        llrs__ = self._readout_llrs[0](s)
                        llrs__ = llrs__[:, :, :, :, : self._num_bits_per_symbol[idx]]
                    else:
                        llrs__ = self._readout_llrs[idx](s)
                    llrs_.append(llrs__)
                llrs.append(llrs_)
                h_hats.append(self._readout_chest(s))

        return llrs, h_hats


class CGNNOFDM(nn.Module):
    r"""
    Wrapper for the neural receiver (CGNN) layer that handles
    OFDM waveforms and the resource grid mapping/demapping.
    Layer also integrates loss function computation.
    """

    def __init__(
        self,
        sys_parameters,
        max_num_tx,
        training,
        num_it=5,
        d_s=32,
        num_units_init=[64],
        num_units_agg=[[64]],
        num_units_state=[[64]],
        num_units_readout=[64],
        layer_demappers=None,
        layer_type_dense="dense",
        layer_type_conv="sepconv",
        layer_type_readout="dense",
        nrx_dtype=torch.float32,
        debug=False,  # Debugging flag for verbose output
        **kwargs,
    ):
        super().__init__()

        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype
        self._debug = debug

        # Debugging flag
        if self._debug:
            print(f"CGNNOFDM initialized with debug mode: {debug}")

        # Number of MCS supported (Modulation and Coding Scheme)
        self._num_mcss_supported = len(sys_parameters.mcs_index)

        # Resource grid initialization
        self._rg = sys_parameters.transmitters[0]._resource_grid

        if self._sys_parameters.mask_pilots:
            print("Masking pilots for pilotless communications.")

        # Variable MCS Masking
        self._mcs_var_mcs_masking = (
            hasattr(self._sys_parameters, "mcs_var_mcs_masking")
            and self._sys_parameters.mcs_var_mcs_masking
        )

        if self._debug:
            print(f"Var-MCS Masking: {self._mcs_var_mcs_masking}")

        # Initialize bits per symbol for MCS schemes
        num_bits_per_symbol = [
            sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol
            for mcs_list_idx in range(self._num_mcss_supported)
        ]

        # Number of receive antennas
        num_rx_ant = sys_parameters.num_rx_antennas

        # Initialize the Core Neural Receiver
        self._cgnn = CGNN(
            num_bits_per_symbol,
            num_rx_ant,
            num_it,
            d_s,
            num_units_init,
            num_units_agg,
            num_units_state,
            num_units_readout,
            training=training,
            layer_type_dense=layer_type_dense,
            layer_type_conv=layer_type_conv,
            layer_type_readout=layer_type_readout,
            var_mcs_masking=self._mcs_var_mcs_masking,
            dtype=nrx_dtype,
        )

        # Resource grid demapper
        self._rg_demapper = ResourceGridDemapper(self._rg, sys_parameters.sm)

        # Loss function initialization
        if training:
            self._bce = nn.BCEWithLogitsLoss(reduction="none")
            self._mse = nn.MSELoss(reduction="none")

        # Precompute positional encoding for pilots with detailed debug outputs
        self._precompute_pilot_positional_encoding()

    def _precompute_pilot_positional_encoding(self):
        """
        Precomputes the positional encoding based on the distance to the nearest pilot
        in both time and frequency, and handles pilot insertion safely.
        """

        # Extract resource grid type (data, pilot, guard, DC)
        rg_type = self._rg.build_type_grid()[:, 0].numpy()  # Consider only one stream
        if self._debug:
            print(f"Resource grid type shape: {rg_type.shape}")

        rg_type_torch = torch.tensor(rg_type)  # Convert to PyTorch tensor

        # Get pilot indices (where value is 1)
        pilot_ind = torch.where(rg_type_torch == 1)

        # Ensure valid pilot indices (within bounds of the resource grid)
        valid_pilot_indices = (pilot_ind[0] < self._rg.fft_size) & (pilot_ind[0] >= 0)
        pilot_ind = pilot_ind[0][valid_pilot_indices]

        if self._debug:
            print(f"Valid pilot indices shape: {pilot_ind.shape}")

        # Convert pilots to PyTorch tensor
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3).numpy()
        pilots_torch = torch.tensor(pilots, dtype=torch.float32)

        # Initialize an empty pilot tensor of the same shape as rg_type
        pilots_only = torch.zeros_like(rg_type_torch, dtype=pilots_torch.dtype)

        # Insert pilots into the grid without out-of-bounds errors
        for i, p_ind in enumerate(pilot_ind):
            if p_ind < pilots_only.shape[0]:  # Check if index is valid
                pilots_only[p_ind] = pilots_torch[i]
            else:
                if self._debug:
                    print(f"Skipping out-of-bounds pilot index: {p_ind}")

        # Scatter pilots into the correct locations
        pilot_ind = torch.where(torch.abs(pilots_only) > 1e-3)

        if self._debug:
            print(f"Pilots inserted: {len(pilot_ind[0])}")

        # Precompute distances to the nearest pilots in time and frequency
        self._compute_pilot_distances(pilot_ind)

    def _compute_pilot_distances(self, pilot_ind):
        """
        Computes the distances to the nearest pilot in both time and frequency dimensions.
        Handles dimension mismatches safely and ensures correct broadcasting.
        """
        if self._debug:
            print(f"Computing pilot distances. Number of pilots: {len(pilot_ind[0])}")

        # Time and frequency indices
        t_ind = torch.arange(self._rg.num_ofdm_symbols)  # 14 OFDM symbols
        f_ind = torch.arange(self._rg.fft_size)  # FFT size, e.g., 76 subcarriers

        # Initialize distance matrices for time and frequency
        num_pilots = len(pilot_ind[0])  # Number of valid pilot positions
        pilots_dist_time = torch.zeros(
            (self._max_num_tx, self._rg.num_ofdm_symbols, self._rg.fft_size, num_pilots)
        )
        pilots_dist_freq = torch.zeros(
            (self._max_num_tx, self._rg.num_ofdm_symbols, self._rg.fft_size, num_pilots)
        )

        # Calculate the distance for each transmitter
        for tx_ind in range(self._max_num_tx):
            for i, p_ind in enumerate(pilot_ind[0]):  # Pilot index is a 1D array
                # Ensure p_ind is compatible with the size of t_ind and f_ind for broadcasting
                if (
                    p_ind < self._rg.num_ofdm_symbols
                ):  # Ensure pilot index is within bounds
                    pilots_dist_time[tx_ind, :, :, i] = (
                        torch.abs(t_ind - p_ind)
                        .unsqueeze(1)
                        .expand(-1, self._rg.fft_size)
                    )
                if p_ind < self._rg.fft_size:  # Ensure pilot index is within bounds
                    pilots_dist_freq[tx_ind, :, :, i] = (
                        torch.abs(f_ind - p_ind)
                        .unsqueeze(0)
                        .expand(self._rg.num_ofdm_symbols, -1)
                    )

        # Normalize the distances for stability
        nearest_pilot_dist_time = pilots_dist_time.min(dim=-1)[0]
        nearest_pilot_dist_time -= nearest_pilot_dist_time.mean(dim=1, keepdim=True)
        std_time = nearest_pilot_dist_time.std(dim=1, keepdim=True)
        nearest_pilot_dist_time = torch.where(
            std_time > 0.0, nearest_pilot_dist_time / std_time, nearest_pilot_dist_time
        )

        nearest_pilot_dist_freq = pilots_dist_freq.min(dim=-1)[0]
        nearest_pilot_dist_freq -= nearest_pilot_dist_freq.mean(dim=2, keepdim=True)
        std_freq = nearest_pilot_dist_freq.std(dim=2, keepdim=True)
        nearest_pilot_dist_freq = torch.where(
            std_freq > 0.0, nearest_pilot_dist_freq / std_freq, nearest_pilot_dist_freq
        )

        # Combine the distances and set it as an instance variable
        nearest_pilot_dist = torch.stack(
            [nearest_pilot_dist_time, nearest_pilot_dist_freq], dim=-1
        )
        self._nearest_pilot_dist = nearest_pilot_dist.permute(0, 2, 1, 3)

        if self._debug:
            print(f"Nearest pilot distance shape: {self._nearest_pilot_dist.shape}")

    @property
    def num_it(self):
        """Number of receiver iterations. No weight sharing is used."""
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        self._cgnn.num_it = val

    def forward(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None):
        if self._training:
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask = inputs
        else:
            y, h_hat_init, active_tx = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = F.one_hot(
                    torch.tensor(mcs_arr_eval[0]), num_classes=self._num_mcss_supported
                ).float()
            else:
                mcs_ue_mask = mcs_ue_mask_eval

        # Core Neural Receiver
        num_tx = active_tx.shape[1]

        if self._sys_parameters.mask_pilots:
            rg_type = self._rg.build_type_grid().unsqueeze(0).expand_as(y)
            y = torch.where(rg_type == 1, torch.zeros_like(y), y)

        y = torch.cat([y.real, y.imag], dim=-1)
        pe = self._nearest_pilot_dist[:num_tx]

        y = y.type(self._nrx_dtype)
        pe = pe.type(self._nrx_dtype)
        if h_hat_init is not None:
            h_hat_init = h_hat_init.type(self._nrx_dtype)
        active_tx = active_tx.type(self._nrx_dtype)

        llrs_, h_hats_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask])

        llrs, h_hats = [], []
        for llrs_, h_hat_ in zip(llrs_, h_hats_):
            h_hat_ = h_hat_.type(torch.float32)
            _llrs_ = []
            for idx in mcs_arr_eval:
                llrs_[idx] = llrs_[idx].type(torch.float32)
                llrs_[idx] = self._rg_demapper(llrs_[idx].unsqueeze(1))
                llrs_[idx] = flatten_last_dims(llrs_[idx], 2)
                if self._layer_demappers is None:
                    llrs_[idx] = llrs_[idx].squeeze(-2)
                else:
                    llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                _llrs_.append(llrs_[idx])

            llrs.append(_llrs_)
            h_hats.append(h_hat_)

        if self._training:
            loss_data = torch.tensor(0.0, dtype=torch.float32)
            for llrs_ in llrs:
                for idx in range(len(mcs_arr_eval)):
                    loss_data_ = self._bce(bits[idx], llrs_[idx])
                    mcs_ue_mask_ = mcs_ue_mask[:, :, idx].unsqueeze(-1)
                    loss_data_ = loss_data_ * mcs_ue_mask_
                    loss_data_ = loss_data_ * active_tx.unsqueeze(-1)
                    loss_data += loss_data_.mean()

            loss_chest = torch.tensor(0.0, dtype=torch.float32)
            if h_hats is not None:
                for h_hat_ in h_hats:
                    if h is not None:
                        loss_chest += self._mse(h, h_hat_)

            loss_chest = loss_chest * active_tx.unsqueeze(-1)
            loss_chest = loss_chest.mean()

            return loss_data, loss_chest
        else:
            return llrs[-1][0], h_hats[-1]


class RemoveNulledSubcarriers:
    r"""RemoveNulledSubcarriers(resource_grid)

    Removes nulled guard and/or DC subcarriers from a resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    Input
    -----
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], complex64
        Full resource grid.

    Output
    ------
    : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], complex64
        Resource grid without nulled subcarriers.
    """

    def __init__(self, resource_grid):
        # Convert the `range` object to a numpy array
        self._sc_ind = np.array(resource_grid.effective_subcarrier_ind)
        print("sc_ind shape:", self._sc_ind.shape)  # (64,) [5,6,7,8...]

    def __call__(self, inputs):  # inputs (64, 1, 1, 1, 16, 1, 76)
        # Assuming 'inputs' is a NumPy array and '_sc_ind' is an integer or array of indices
        result = np.take(inputs, self._sc_ind, axis=-1)  # (64, 1, 1, 1, 16, 1, 64)
        return result


import os


class NearestNeighborInterpolator:
    # pylint: disable=line-too-long
    r"""NearestNeighborInterpolator(pilot_pattern)


    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, pilot_pattern):
        # super().__init__()

        assert (
            pilot_pattern.num_pilot_symbols > 0
        ), """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)  # (1, 2, 14, 64)
        mask_shape = (
            mask.shape
        )  # Store to reconstruct the original shape (1, 2, 14, 64)
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))  # (2, 14, 64)

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots  # (1, 2, 128)
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])  # (2, 128)

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))  # 64
        assert (
            max_num_zero_pilots < pilots.shape[-1]
        ), """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)  # (2, 14, 64)
        for a in range(gather_ind.shape[0]):  # For each pilot pattern...
            i_p, j_p = np.where(mask[a])  # ...determine the pilot indices

            for i in range(mask_shape[-2]):  # Iterate over...
                for j in range(mask_shape[-1]):  # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i - i_p) + np.abs(j - j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a]) == 0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        # self._gather_ind = tf.reshape(gather_ind, mask_shape)
        self._gather_ind = np.reshape(
            gather_ind, mask_shape
        )  # _gather_ind: (1, 2, 14, 64)

        # Create 'data/' directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        np.save("data/inter_gather_ind.npy", self._gather_ind)

    def mygather(self, inputs, method="tf"):
        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        if method == "tf":
            outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2)
        # batch_dims: An optional parameter that specifies the number of batch dimensions. It controls how many leading dimensions are considered as batch dimensions.
        elif method == "np":
            result = inputs.copy()
            # Gather along each dimension
            # for dim in range(batch_dims, len(inputs.shape)): #2-6
            #     result = np.take(result, indices, axis=dim)

            # gather_ind_nobatch = indices[0, 0] #ignore first two dimensions as batch (14, 64)
            # result = np.take(result, gather_ind_nobatch, axis=2) #(1, 2, 14, 64, 2, 1, 16)
            gather_ind_nobatch = self._gather_ind[
                0, 0
            ]  # ignore first two dimensions as batch (14, 64)
            result1 = np.take(
                result, gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            gather_ind_nobatch = self._gather_ind[
                0, 1
            ]  # ignore first two dimensions as batch (14, 64)
            outputs = np.take(
                result, gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            outputs[0, 0, :, :, :, :, :] = result1[0, 0, :, :, :, :, :]
        else:  # Wrong result
            # outputs = np.take(inputs, self._gather_ind, axis=2, mode='wrap') #(1, 2, 1, 2, 14, 64, 2, 1, 16), _gather_ind: (1, 2, 14, 64)
            # outputs: (1, 2, 14, 64, 2, 1, 16)
            self._gather_ind_nobatch = self._gather_ind[
                0, 0
            ]  # ignore first two dimensions as batch (14, 64)
            outputs = np.take(
                inputs, self._gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            np.save("data/outputs_inter.npy", outputs)
            # outputs = inputs[:, :, self._gather_ind, :, :, :] #(1, 2, 1, 2, 14, 64, 2, 1, 16)
            # Perform the gathe
            # axis = 2
            # batch_dims = 2
            # outputs = np.take_along_axis(inputs, self._gather_ind, axis=axis, batch_dims=batch_dims)
        return outputs

    def _interpolate(self, inputs):
        # inputs has shape: (1, 2, 128, 2, 1, 16)
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        # perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0)
        # inputs = tf.transpose(inputs, perm)
        perm = np.roll(
            np.arange(np.ndim(inputs)), -3, 0
        )  # shift the dimensions. (2, 1, 16, 1, 2, 128)
        inputs = np.transpose(inputs, perm)  # (1, 2, 128, 2, 1, 16)

        # np.save('inputs_inter.npy', inputs)
        outputs = self.mygather(inputs)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        # perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        # outputs = tf.transpose(outputs, perm)
        perm = np.roll(np.arange(np.ndim(outputs)), 3, 0)  # [4, 5, 6, 0, 1, 2, 3]
        outputs = np.transpose(outputs, perm)  # (2, 1, 16, 1, 2, 14, 64)

        return outputs

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


def myexpand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a ``tensor`` starting at
    ``axis``, so that so that the rank of the resulting tensor has rank
    ``target_rank``. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If ``target_rank`` is smaller than the rank of ``tensor``,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of ``tensor``. Given a ``tensor`` of `D` dimensions,
               ``axis`` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as ``tensor``, with
        ``target_rank``- rank(``tensor``) additional dimensions inserted at the
        index specified by ``axis``.
        If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
    """
    # num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    num_dims = np.maximum(target_rank - tensor.ndim, 0)  # difference in rank, >0 7
    # Adds multiple length-one dimensions to a tensor.
    # It inserts ``num_dims`` dimensions of length one starting from the dimension ``axis``
    # output = insert_dims(tensor, num_dims, axis)
    rank = tensor.ndim  # 1
    axis = axis if axis >= 0 else rank + axis + 1  # 0
    # shape = tf.shape(tensor)
    shape = np.shape(tensor)  # (76,)
    new_shape = np.concatenate(
        [shape[:axis], np.ones([num_dims], np.int32), shape[axis:]], 0
    )  # (8,) array([ 1.,  1.,  1.,  1.,  1.,  1.,  1., 76.])
    # new_shape = tf.concat([shape[:axis],
    #                        tf.ones([num_dims], tf.int32),
    #                        shape[axis:]], 0)
    # output = tf.reshape(tensor, new_shape)
    new_shape = new_shape.astype(np.int32)
    output = np.reshape(tensor, new_shape)  # (76,)

    return output  # (1, 1, 1, 1, 1, 1, 1, 76)


class MyLSChannelEstimatorNP:
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(
        self, resource_grid, interpolation_type="nn", interpolator=None, **kwargs
    ):
        # super().__init__(dtype=dtype, **kwargs)

        # assert isinstance(resource_grid, ResourceGrid),\
        #     "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern

        # added test code
        mask = np.array(self._pilot_pattern.mask)  # (1, 2, 14, 64)
        mask_shape = mask.shape  # Store to reconstruct the original shape
        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = self._pilot_pattern.pilots  # (1, 2, 128)
        print(mask_shape)  # (1, 2, 14, 64)
        # print('mask:', mask[0,0,0,:]) #all 0
        # print('pilots:', pilots[0,0,:]) #(1, 2, 128) -0.99999994-0.99999994j 0.        +0.j          0.99999994+0.99999994j
        # 0.99999994-0.99999994j  0.        +0.j          0.99999994-0.99999994j
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in [
            "nn",
            "lin",
            "lin_time_avg",
            None,
        ], "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        # if self._interpolation_type == "nn":
        self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        # elif self._interpolation_type == "lin":
        #     self._interpol = LinearInterpolator(self._pilot_pattern)
        # elif self._interpolation_type == "lin_time_avg":
        #     self._interpol = LinearInterpolator(self._pilot_pattern,
        #                                         time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols  # 128
        mask = flatten_last_dims(self._pilot_pattern.mask)  # (1, 2, 896)
        # np.save('mask.npy', mask)
        # pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING") #(1, 2, 896)
        ##np.argsort is small to bigger (index of 0s first, index of 1s later), add [..., ::-1] to flip the results from bigger to small (index 1s first, index 0s later)
        pilot_ind = np.argsort(mask, axis=-1)[
            ..., ::-1
        ]  # (1, 2, 896) reverses the order of the indices along the last axis
        # select num_pilot_symbols, i.e., get all index of 1s in mask, due to the np.argsort(small to bigger), the order for these 1s index is not sorted
        self._pilot_ind = pilot_ind[..., :num_pilot_symbols]  # (1, 2, 128)
        # add sort again for these 1s index (small index to bigger)
        # analysis in tfnumpy.py
        self._pilot_ind = np.sort(self._pilot_ind)
        print(self._pilot_ind)

    def estimate_at_pilot_locations(self, y_pilots, no):

        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        # h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)
        # h_ls = np.divide(y_pilots, self._pilot_pattern.pilots) #(2, 1, 16, 1, 2, 128)
        # h_ls = np.nan_to_num(h_ls) #replaces NaN (Not-a-Number) values with zeros.

        h_ls = np.divide(
            y_pilots,
            self._pilot_pattern.pilots,
            out=np.zeros_like(y_pilots),
            where=self._pilot_pattern.pilots != 0,
        )
        # h_ls: (2, 1, 16, 1, 2, 128)
        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        # no = expand_to_rank(no, tf.rank(h_ls), -1)
        # no = myexpand_to_rank(no, h_ls.ndim, -1) #(1, 1, 1, 1, 1, 1)

        # Expand rank of pilots for broadcasting
        # pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)
        pilots = myexpand_to_rank(
            self._pilot_pattern.pilots, h_ls.ndim, 0
        )  # (1, 1, 1, 1, 2, 128)

        # Compute error variance, broadcastable to the shape of h_ls
        # err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)
        pilotssquare = np.abs(pilots) ** 2
        # err_var = np.divide(no, pilotssquare)
        # err_var = np.nan_to_num(err_var) #replaces NaN (Not-a-Number) values with zeros. (1, 1, 1, 1, 2, 128)
        no_array = np.full(pilots.shape, no, dtype=np.float32)  # (1, 1, 1, 1, 2, 128)
        err_var = np.divide(
            no_array, pilotssquare, out=np.zeros_like(no_array), where=pilotssquare != 0
        )  # (1, 1, 1, 1, 2, 128)

        return h_ls, err_var

    # def call(self, inputs):
    def __call__(self, inputs):

        y, no = inputs  # y: (64, 1, 1, 14, 76) complex64
        y = to_numpy(y)  # (2, 1, 16, 14, 76)
        no = np.array(no, dtype=np.float32)

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y)  # (2, 1, 16, 14, 64) complex64

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)  # (2, 1, 16, 896)
        # plt.figure()
        # plt.plot(np.real(y_eff_flat[0,0,0,:]))
        # plt.plot(np.imag(y_eff_flat[0,0,0,:]))
        # plt.title('y_eff_flat')

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        # y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)
        # y_pilots = y_eff_flat[self._pilot_ind] #(2, 1, 16, 896)
        # y_pilots = np.take(y_eff_flat, self._pilot_ind, axis=-1) #y_eff_flat:(2, 1, 16, 896), _pilot_ind:(1, 2, 128) => y_pilots(2, 1, 16, 1, 2, 128)
        # Gather elements from y_eff_flat based on pilot_ind
        y_pilots = y_eff_flat[..., self._pilot_ind]  # (2, 1, 16, 1, 2, 128)

        # plt.figure()
        # plt.plot(np.real(y_pilots[0,0,0,0,0,:]))
        # plt.plot(np.imag(y_pilots[0,0,0,0,0,:]))
        # plt.title('y_pilots')
        # np.save('y_eff_flat.npy', y_eff_flat)
        # np.save('pilot_ind.npy', self._pilot_ind)
        # np.save('y_pilots.npy', y_pilots)

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(
            y_pilots, no
        )  # y_pilots: (2, 1, 16, 1, 2, 128), h_hat:(2, 1, 16, 1, 2, 128)
        # np.save('h_hat_pilot.npy', h_hat) #(2, 1, 16, 1, 2, 128)
        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(
                h_hat, err_var
            )  # h_hat: (2, 1, 16, 1, 2, 128)=>
            # np.save('h_hat_inter.npy', h_hat)
            # err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))
            err_var = np.maximum(err_var, 0)

        return h_hat, err_var


class NeuralPUSCHReceiver(nn.Module):
    def __init__(self, sys_parameters, training=False, **kwargs):

        super().__init__()
        print("NeuralPUSCHReceiver init")

        self._sys_parameters = sys_parameters
        self._training = False

        # init transport block encoder/decoder
        self._tb_encoders = []
        self._tb_decoders = []

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        rg = sys_parameters.transmitters[0]._resource_grid
        self._pilots = rg.pilot_pattern.pilots  # Extract pilots from resource grid
        self._num_mcss_supported = len(sys_parameters.mcs_index)

        for mcs_list_idx in range(self._num_mcss_supported):
            self._tb_encoders.append(
                self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder
            )

            self._tb_decoders.append(
                TBDecoder(
                    self._tb_encoders[mcs_list_idx],
                    num_bp_iter=sys_parameters.num_bp_iter,
                    cn_type=sys_parameters.cn_type,
                )
            )

        # Precoding matrix to post-process the ground-truth channel when training
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = torch.ones(
                [sys_parameters.max_num_tx, sys_parameters.num_antenna_ports, 1],
                dtype=torch.complex64,
            )

        # LS channel estimator
        self.rg = sys_parameters.transmitters[0]._resource_grid
        pc = sys_parameters.pusch_configs[0][0]
        # Initialize the numpy-based LS channel estimator
        self._ls_est_np = MyLSChannelEstimatorNP(self.rg, interpolation_type="nn")
        # self._ls_est = PUSCHLSChannelEstimator(
        #     resource_grid=rg,
        #     dmrs_length=pc.dmrs.length,
        #     dmrs_additional_position=pc.dmrs.additional_position,
        #     num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
        #     interpolation_type="nn",
        # )

        rg_type = rg.build_type_grid()[:, 0].numpy()
        rg_type_torch = torch.tensor(rg_type)
        pilot_ind = torch.where(rg_type_torch == 1)
        self._pilot_ind = pilot_ind[0].numpy()

        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
            self._layer_demappers.append(
                LayerDemapper(
                    self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                    sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol,
                )
            )

        self._neural_rx = CGNNOFDM(
            sys_parameters,
            max_num_tx=sys_parameters.max_num_tx,
            training=training,
            num_it=sys_parameters.num_nrx_iter,
            d_s=sys_parameters.d_s,
            num_units_init=sys_parameters.num_units_init,
            num_units_agg=sys_parameters.num_units_agg,
            num_units_state=sys_parameters.num_units_state,
            num_units_readout=sys_parameters.num_units_readout,
            layer_demappers=self._layer_demappers,
            layer_type_dense=sys_parameters.layer_type_dense,
            layer_type_conv=sys_parameters.layer_type_conv,
            layer_type_readout=sys_parameters.layer_type_readout,
            dtype=sys_parameters.nrx_dtype,
        )

    def estimate_channel(self, y, num_tx, no):
        """
        Estimate channel using the numpy-based LS method.
        y has shape [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        num_tx is the number of transmitters.
        no is the noise variance.
        """
        print("NeuralPUSCHReceiver estimate_channel")
        print("y", y.shape)
        print("num_tx", num_tx)
        print("no", no.shape)
        print("self._sys_parameters.initial_chest", self._sys_parameters.initial_chest)

        if self._sys_parameters.initial_chest == None:
            if self._sys_parameters.mask_pilots:
                raise ValueError(
                    "Cannot use initial channel estimator if pilots are masked."
                )

            # Convert PyTorch tensor `y` to NumPy for compatibility with MyLSChannelEstimatorNP
            y_numpy = y.cpu().numpy()  # Assuming `y` is a PyTorch tensor
            no_numpy = (
                no.cpu().numpy() if isinstance(no, torch.Tensor) else no
            )  # Handle `no` in case it's a tensor

            # Use the numpy-based LS estimator
            print(self.rg, self._pilots)
            h_hat_numpy, err_var_numpy = self._ls_est_np()
            print("estimated channel")
            # Convert the results back to PyTorch tensors
            h_hat = torch.from_numpy(h_hat_numpy).to(y.device)
            err_var = torch.from_numpy(err_var_numpy).to(y.device)

            print(h_hat.shape, err_var.shape)
            return h_hat, err_var

    def preprocess_channel_ground_truth(self, h):
        # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_effective_subcarriers]

        # Assume only one rx
        h = torch.squeeze(
            h, dim=1
        )  # [batch_size, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Reshape h
        h = h.permute(
            0, 2, 5, 4, 1, 3
        )  # [batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_tx_ant]

        # Multiply by precoding matrices to compute effective channels
        w = self._precoding_mat.unsqueeze(0).unsqueeze(
            2
        )  # [1, num_tx, 1, 1, num_tx_ant, 1]
        h = torch.matmul(h, w).squeeze(
            -1
        )  # [batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant]

        # Complex-to-real
        h = torch.cat(
            [torch.real(h), torch.imag(h)], dim=-1
        )  # [batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers, 2*num_rx_ant]

        return h

    def forward(self, inputs, no, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        """
        Apply neural receiver.
        """
        self._training = False
        print("NeuralPUSCHReceiver forward")
        print(self._training)
        if self._training:
            # In training mode, we expect 5 inputs
            y, active_tx, b, h, mcs_ue_mask = inputs

            # Re-encode bits in training mode to generate labels
            if len(mcs_arr_eval) == 1 and not isinstance(b, list):
                b = [b]  # generate new list if b is not provided as list

            bits = []
            for idx in range(len(mcs_arr_eval)):
                bits.append(
                    self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(
                        b[idx]
                    )
                )

            # Initial channel estimation
            num_tx = active_tx.shape[1]
            h_hat = self.estimate_channel(y, num_tx)

            # Reshaping `h` to the expected shape and apply precoding matrices
            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            # Apply neural receiver and return loss
            losses = self._neural_rx(
                (y, h_hat, active_tx, bits, h, mcs_ue_mask), mcs_arr_eval
            )
            return losses

        else:
            # In evaluation, we expect only 2 inputs
            print("NeuralPUSCHReceiver forward else")
            y, active_tx = inputs
            num_tx = active_tx.shape[1]
            print("Flag1")
            h_hat, err_var = self.estimate_channel(y, num_tx, no)

            # Now you can access h_hat.shape safely
            print(h_hat.shape, h_hat)

            # Call the neural receiver and get llr and h_hat_refined
            llr, h_hat_refined = self._neural_rx(
                (y, h_hat, active_tx),
                [mcs_arr_eval[0]],
                mcs_ue_mask_eval=mcs_ue_mask_eval,
            )

            # Apply TBDecoding
            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            # Return the decoded bits, refined channel estimates, and CRC status
            return b_hat, h_hat_refined, h_hat, tb_crc_status


################################
## ONNX Layers / Wrapper
################################
# The following layers provide an adapter to the Aerial PUSCH pipeline
# the code is only relevant for for ONNX/TensorRT exports but can be ignored
# for Sionna-based simulations.


class NRPreprocessing(nn.Module):
    def __init__(self, num_tx, **kwargs):
        super().__init__()
        self._num_tx = num_tx
        self._num_res_per_prb = 12  # fixed in 5G

    def _focc_removal(self, h_hat):
        shape = [-1, 2]
        s = h_hat.shape
        new_shape = [*s[:3], *shape]
        h_hat = h_hat.view(new_shape)
        h_hat = h_hat.sum(dim=-1, keepdim=True) / 2.0
        h_hat = h_hat.repeat(1, 1, 1, 1, 2)
        new_shape = [*s[:3], -1]
        h_hat = h_hat.view(new_shape)
        return h_hat

    def _calculate_nn_indices(
        self, dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
    ):
        re_pos = torch.meshgrid(
            torch.arange(self._num_res_per_prb), torch.arange(num_ofdm_symbols)
        )
        re_pos = torch.stack(re_pos, dim=-1).view(-1, 1, 2)

        pes, nn_idxs = [], []
        for tx_idx in range(self._num_tx):
            p_idx = torch.meshgrid(dmrs_subcarrier_pos[tx_idx], dmrs_ofdm_pos[tx_idx])
            pilot_pos = torch.stack(p_idx, dim=-1).view(1, -1, 2)
            diff = torch.abs(re_pos - pilot_pos)
            dist = diff.sum(dim=-1)

            nn_idx = dist.argmin(dim=1).view(
                1, 1, num_ofdm_symbols, self._num_res_per_prb
            )
            pe = diff.min(dim=1)[0].view(1, num_ofdm_symbols, self._num_res_per_prb, 2)
            pe = pe.permute(0, 2, 1, 3)

            pe = pe.float()
            pe_ = (pe[..., 1:2] - pe[..., 1:2].mean()) / (pe[..., 1:2].std() + 1e-8)
            p = [pe_]

            pe_ = (pe[..., 0:1] - pe[..., 0:1].mean()) / (pe[..., 0:1].std() + 1e-8)
            p.append(pe_)

            pe = torch.cat(p, dim=-1)
            pes.append(pe)
            nn_idxs.append(nn_idx)

        pe = torch.cat(pes, dim=0).repeat(1, num_prbs, 1, 1)
        nn_idx = torch.cat(nn_idxs, dim=0)
        return nn_idx, pe

    def _nn_interpolation(
        self, h_hat, num_ofdm_symbols, dmrs_ofdm_pos, dmrs_subcarrier_pos
    ):
        num_pilots_per_dmrs = dmrs_subcarrier_pos.shape[1]
        num_prbs = h_hat.shape[-1] // (num_pilots_per_dmrs * dmrs_ofdm_pos.shape[-1])

        h_hat = h_hat.view(-1, num_pilots_per_dmrs, num_prbs, *h_hat.shape[1:])
        h_hat = h_hat.permute(0, 1, 3, 2, 4, 5).reshape_as(h_hat)

        h_hat = h_hat.unsqueeze(1).unsqueeze(4)
        h_hat = h_hat.permute(torch.roll(torch.arange(h_hat.dim()), shifts=-3))

        ls_nn_ind, pe = self._calculate_nn_indices(
            dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
        )

        h_hat_prb = h_hat.view(*h_hat.shape[:2], num_prbs, -1, *h_hat.shape[3:])
        outputs = h_hat_prb.gather(2, ls_nn_ind.unsqueeze(2).expand_as(h_hat_prb))
        outputs = outputs.permute(0, 1, 2, 4, 3, 5, 6, 7)

        outputs = outputs.view(
            -1,
            *outputs.shape[1:3],
            num_prbs * self._num_res_per_prb,
            *outputs.shape[5:],
        )
        outputs = outputs.permute(torch.roll(torch.arange(outputs.dim()), shifts=3))

        h_hat = outputs
        return h_hat, pe

    def forward(self, inputs):
        y, h_hat_ls, dmrs_ofdm_pos, dmrs_subcarrier_pos = inputs
        num_ofdm_symbols = y.shape[2]

        h_hat_ls = h_hat_ls.permute(0, 3, 2, 1)
        h_hat_ls = self._focc_removal(h_hat_ls)

        h_hat, pe = self._nn_interpolation(
            h_hat_ls, num_ofdm_symbols, dmrs_ofdm_pos, dmrs_subcarrier_pos
        )

        h_hat = h_hat[:, 0, :, : self._num_tx, 0]
        h_hat = h_hat.permute(0, 2, 4, 3, 1)
        return [h_hat, pe]


class NeuralReceiverONNX(nn.Module):
    r"""
    Wraps the 5G NR neural receiver in an ONNX compatible format.
    """

    def __init__(
        self,
        num_it,
        d_s,
        num_units_init,
        num_units_agg,
        num_units_state,
        num_units_readout,
        num_bits_per_symbol,
        layer_type_dense,
        layer_type_conv,
        layer_type_readout,
        nrx_dtype,
        num_tx,
        num_rx_ant,
        **kwargs,
    ):
        super().__init__()

        assert len(num_units_agg) == num_it and len(num_units_state) == num_it

        self._num_tx = num_tx  # Assuming one stream per user

        ####################################################
        # Detector
        ####################################################
        self._cgnn = CGNN(
            [num_bits_per_symbol],  # No support for mixed MCS
            num_rx_ant,
            num_it,
            d_s,
            num_units_init,
            num_units_agg,
            num_units_state,
            num_units_readout,
            layer_type_dense=layer_type_dense,
            layer_type_conv=layer_type_conv,
            layer_type_readout=layer_type_readout,
            dtype=nrx_dtype,
        )

        self._preprocessing = NRPreprocessing(self._num_tx)

    @property
    def num_it(self):
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (
            val <= len(self._cgnn._iterations)
        ), "Invalid number of iterations"
        self._cgnn.num_it = val

    def forward(self, inputs):
        (
            y_real,
            y_imag,
            h_hat_real,
            h_hat_imag,
            dmrs_port_mask,
            dmrs_ofdm_pos,
            dmrs_subcarrier_pos,
        ) = inputs

        # Concatenate real and imaginary parts for y and h_hat
        y = torch.cat((y_real, y_imag), dim=-1)
        h_hat_p = torch.cat((h_hat_real, h_hat_imag), dim=-1)

        # Nearest neighbor interpolation of channel estimates
        h_hat, pe = self._preprocessing(
            (y, h_hat_p, dmrs_ofdm_pos, dmrs_subcarrier_pos)
        )

        # Dummy MCS mask (no support for mixed MCS)
        mcs_ue_mask = torch.ones((1, 1, 1), dtype=torch.float32)

        # Run NRX (Neural Receiver)
        llr, h_hat = self._cgnn([y, pe, h_hat, dmrs_port_mask, mcs_ue_mask])

        # CGNN returns a list of results for each iteration
        # Use the results from the final iteration
        llr = llr[-1][0]  # Take LLRs of first MCS (no support for mixed MCS)
        h_hat = h_hat[-1]

        # Cast back to float32 if NRX uses quantization
        llr = llr.float()
        h_hat = h_hat.float()

        # Reshape LLRs in Aerial format
        llr = llr.permute(0, 4, 1, 2, 3)  # Change axis ordering to match the format
        llr = -1.0 * llr  # Sionna defines LLRs with different sign

        return llr, h_hat
