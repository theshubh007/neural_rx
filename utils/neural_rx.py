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
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Layer
import torch
from torch import nn
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, expand_to_rank
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator


class ResourceGridWrapper:
    def __init__(self, resource_grid):
        self._resource_grid = resource_grid

    def build_type_grid(self):
        return torch.from_numpy(self._resource_grid.build_type_grid().numpy())

    @property
    def pilot_pattern(self):
        return PilotPatternWrapper(self._resource_grid.pilot_pattern)


class PilotPatternWrapper:
    def __init__(self, pilot_pattern):
        self._pilot_pattern = pilot_pattern

    @property
    def pilots(self):
        return torch.from_numpy(self._pilot_pattern.pilots.numpy())

    @property
    def mask(self):
        return torch.from_numpy(self._pilot_pattern.mask.numpy())

    @property
    def num_data_symbols(self):
        return self._pilot_pattern.num_data_symbols


class ResourceGridWrapper:
    def __init__(self, resource_grid):
        self._resource_grid = resource_grid
        self._pilot_pattern = PilotPatternWrapper(resource_grid.pilot_pattern)

    def build_type_grid(self):
        return torch.from_numpy(self._resource_grid.build_type_grid().numpy())

    @property
    def pilot_pattern(self):
        return self._pilot_pattern

    @property
    def num_ofdm_symbols(self):
        return self._resource_grid.num_ofdm_symbols

    @property
    def fft_size(self):
        return self._resource_grid.fft_size


class StateInit(nn.Module):
    """
    Network initializing the state tensor for each user.

    The network consists of len(num_units) hidden blocks, each block
    consisting of:
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity.

    Parameters
    -----------
    d_s : int
        Size of the state vector
    num_units : list of int
        Number of kernels for the hidden layers of the MLP.
    layer_type: str | "sepconv" | "conv"
        Defines which Convolutional layers are used. Will be either
        SeparableConv2d or Conv2d.

    Input
    ------
    (y, pe, h_hat)
    Tuple:
    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], torch.Tensor
        The received OFDM resource grid after cyclic prefix removal and FFT.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.
    h_hat : None or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                     2*num_rx_ant], torch.Tensor
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Initial state tensor for each user.
    """

    def __init__(self, d_s, num_units, layer_type="sepconv", dtype=torch.float32):
        super().__init__()
        print("flag: StateInit")
        print(layer_type)
        if layer_type == "sepconv" or layer_type == "separable_conv2d":
            layer = lambda in_c, out_c: nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
                nn.Conv2d(in_c, out_c, kernel_size=1),
            )
        elif layer_type == "conv":
            layer = lambda in_c, out_c: nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        else:
            raise NotImplementedError("Unknown layer_type selected.")
        print("flag2: StateInit")
        # Hidden blocks
        self._hidden_conv = nn.ModuleList()
        in_channels = None  # Will be set in forward pass
        for n in num_units:
            conv = nn.Sequential(
                layer(in_channels, n) if in_channels else nn.Identity(), nn.ReLU()
            )
            self._hidden_conv.append(conv)
            in_channels = n

        # Output block
        self._output_conv = layer(in_channels, d_s)

    def forward(self, inputs):
        y, pe, h_hat = inputs

        batch_size = y.shape[0]
        num_tx = pe.shape[1]

        # Stack the inputs
        y = y.unsqueeze(1).repeat(1, num_tx, 1, 1, 1)
        y = flatten_dims(y, 2, 0)

        pe = pe.repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)

        # ignore h_hat if no channel estimate is provided
        if h_hat is not None:
            h_hat = flatten_dims(h_hat, 2, 0)
            z = torch.cat([y, pe, h_hat], dim=-1)
        else:
            z = torch.cat([y, pe], dim=-1)

        # Apply the neural network
        z = z.permute(0, 3, 1, 2)  # [batch*num_tx, channels, height, width]
        for conv in self._hidden_conv:
            z = conv(z)
        z = self._output_conv(z)
        z = z.permute(0, 2, 3, 1)  # [batch*num_tx, height, width, channels]

        # Unflatten
        s0 = split_dim(z, [batch_size, num_tx], 0)

        return s0  # Initial state of every user


class AggregateUserStates(nn.Module):
    """
    For every user n, aggregate the states of all the other users n' != n.

    An MLP is applied to every state before aggregating.
    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `d_s` units.

    The input `active_tx` provides a mask of active users and non-active users
    will be ignored in the aggregation.

    Parameters
    -----------
    d_s : int
        Size of the state vector

    num_units : list of int
        Number of units for the hidden layers.

    layer_type: str | "dense"
        Defines which Dense layers are used.

    Input
    ------
    (s, active_tx)
    Tuple:

    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Size of the state vector.

    active_tx: [batch_size, num_tx], torch.Tensor
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        For every user `n`, aggregate state of the other users, i.e.,
        sum(s, axis=1) - s[:,n,:,:,:]
    """

    def __init__(self, d_s, num_units, layer_type="dense", dtype=torch.float32):
        super().__init__()
        print("flag: AggregateUserStates")
        print(layer_type)

        if layer_type not in ["dense", "linear"]:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = nn.ModuleList()
        for n in num_units:
            self._hidden_layers.append(
                nn.Sequential(nn.Linear(d_s, n, dtype=dtype), nn.ReLU())
            )
        self._output_layer = nn.Linear(
            num_units[-1] if num_units else d_s, d_s, dtype=dtype
        )

    def forward(self, inputs):
        """
        s, active_tx = inputs

        s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
            torch.Tensor
            State tensor.
        active_tx: [batch_size, num_tx], torch.Tensor
            Active user mask.
        """

        s, active_tx = inputs

        # Process s
        # Output : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        sp = s
        for layer in self._hidden_layers:
            sp = layer(sp)
        sp = self._output_layer(sp)

        # Aggregate all states
        # [batch_size, 1, num_subcarriers, num_ofdm_symbols, d_s]
        # mask non active users
        active_tx = expand_to_rank(active_tx, sp.dim(), axis=-1)
        sp = torch.mul(sp, active_tx)

        # aggregate and remove self-state
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        a = torch.sum(sp, dim=1, keepdim=True) - sp

        # scale by number of active users
        p = torch.sum(active_tx, dim=1, keepdim=True) - 1.0
        p = torch.relu(p)  # clip negative values to ignore non-active user

        # avoid 0 for single active user
        p = torch.where(
            p == 0.0, torch.tensor(1.0, device=p.device), 1.0 / p.clamp(min=1e-10)
        )

        # and scale states by number of aggregated users
        a = torch.mul(a, p)

        return a


class UpdateState(nn.Module):
    """
    Updates the state tensor.

    The network consists of len(num_units) hidden blocks, each block i
    consisting of:
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity.

    The network ends with a skip connection with the state.

    Parameters
    -----------
    d_s : int
        Size of the state vector.
    num_units : list of int
        Number of kernels for the hidden separable convolutional layers.
    layer_type: str | "sepconv" | "conv"
        Defines which Convolutional layers are used. Will be either
        SeparableConv2d or Conv2d.

    Input
    ------
    (s, a, pe)
    Tuple:
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Size of the state vector.
    a : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Aggregated states from other users.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Updated channel state vector.
    """

    def __init__(self, d_s, num_units, layer_type="sepconv", dtype=torch.float32):
        super().__init__()
        print("flag: UpdateState")
        print(layer_type)
        if layer_type == "sepconv" or layer_type == "separable_conv2d":
            layer = lambda in_c, out_c: nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
                nn.Conv2d(in_c, out_c, kernel_size=1),
            )
        elif layer_type == "conv":
            layer = lambda in_c, out_c: nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = nn.ModuleList()
        in_channels = 2 * d_s + 2  # Initial input channels
        for n in num_units:
            conv = nn.Sequential(layer(in_channels, n), nn.ReLU())
            self._hidden_conv.append(conv)
            in_channels = n

        # Output block
        self._output_conv = layer(in_channels, d_s)

    def forward(self, inputs):
        s, a, pe = inputs

        batch_size, num_tx = s.shape[:2]

        # Stack the inputs
        pe = pe.repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)
        s = flatten_dims(s, 2, 0)
        a = flatten_dims(a, 2, 0)

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 2*d_s + 2]
        z = torch.cat([a, s, pe], dim=-1)

        # Apply the neural network
        z = z.permute(0, 3, 1, 2)  # [batch*num_tx, channels, height, width]
        for conv in self._hidden_conv:
            z = conv(z)
        z = self._output_conv(z)
        z = z.permute(0, 2, 3, 1)  # [batch*num_tx, height, width, channels]

        # Skip connection
        z = z + s

        # Unflatten
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new  # Update tensor state for each user


class CGNNIt(nn.Module):
    """
    Implements an iteration of the CGNN detector.

    Consists of two stages: State aggregation followed by state update.

    Parameters
    -----------
    d_s : int
        Size of the state vector.
    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network
    num_units_state_update : list of int
        Number of kernel for the hidden separable convolutional layers of the
        state-update network
    layer_type_dense: str | "dense"
        Layer type of Dense layers. Dense is used for state aggregation.
    layer_type_conv: str | "sepconv" | "conv"
        Layer type of convolutional layers. CNNs are used for state updates.
    dtype: torch.dtype
        Dtype of the layer.

    Input
    ------
    (s, pe, active_tx)
    Tuple:
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Size of the state vector.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.
    active_tx: [batch_size, num_tx], torch.Tensor
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Updated channel state vector.
    """

    def __init__(
        self,
        d_s,
        num_units_agg,
        num_units_state_update,
        layer_type_dense="dense",
        layer_type_conv="sepconv",
        dtype=torch.float32,
    ):
        super().__init__()

        # Layer for state aggregation
        print("flag: CGNNIt")
        print(layer_type_dense)
        self._state_aggreg = AggregateUserStates(
            d_s, num_units_agg, layer_type_dense, dtype=dtype
        )

        # State update
        self._state_update = UpdateState(
            d_s, num_units_state_update, layer_type_conv, dtype=dtype
        )

    def forward(self, inputs):
        s, pe, active_tx = inputs

        # User state aggregation
        a = self._state_aggreg((s, active_tx))

        # State update
        s_new = self._state_update((s, a, pe))

        return s_new


class ReadoutLLRs(nn.Module):
    """
    Network computing LLRs from the state vectors.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.

    Parameters
    -----------
    num_bits_per_symbol : int
        Number of bits per symbol.
    num_units : list of int
        Number of units for the hidden layers.
    layer_type: str | "dense"
        Defines which type of Dense layers are used.
    dtype: torch.dtype
        Dtype of the layer.

    Input
    ------
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Data state.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
       num_bits_per_symbol], torch.Tensor
        LLRs for each bit of each stream.
    """

    def __init__(
        self, num_bits_per_symbol, num_units, layer_type="dense", dtype=torch.float32
    ):
        super().__init__()

        if layer_type != "dense" and layer_type != "linear":
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = nn.ModuleList()
        for n in num_units:
            self._hidden_layers.append(
                nn.Sequential(nn.Linear(n, n, dtype=dtype), nn.ReLU())
            )

        self._output_layer = nn.Linear(
            num_units[-1] if num_units else num_bits_per_symbol,
            num_bits_per_symbol,
            dtype=dtype,
        )

    def forward(self, s):
        # Input of the MLP
        z = s
        # Apply MLP
        for layer in self._hidden_layers:
            z = layer(z)
        llr = self._output_layer(z)

        return llr  # LLRs on the transmitted bits


class ReadoutChEst(nn.Module):
    """
    Network computing channel estimate.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.

    Parameters
    -----------
    num_rx_ant : int
        Number of receive antennas.
    num_units : list of int
        Number of units for the hidden layers.
    layer_type: str | "dense"
        Defines which Dense layers are used.
    dtype: torch.dtype
        Data type of the layer.

    Input
    ------
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Data state.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], torch.Tensor
        Channel estimate for each stream.
    """

    def __init__(
        self, num_rx_ant, num_units, layer_type="sepconv", dtype=torch.float32
    ):
        super().__init__()

        if layer_type != "dense" and layer_type != "linear":
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = nn.ModuleList()
        input_dim = None  # Will be set in forward pass
        for n in num_units:
            self._hidden_layers.append(
                nn.Sequential(
                    (
                        nn.Linear(input_dim, n, dtype=dtype)
                        if input_dim
                        else nn.Identity()
                    ),
                    nn.ReLU(),
                )
            )
            input_dim = n

        self._output_layer = nn.Linear(input_dim, 2 * num_rx_ant, dtype=dtype)

    def forward(self, s):
        """
        s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
            State vector
        """
        # Input of the MLP
        z = s
        # Apply MLP
        for layer in self._hidden_layers:
            z = layer(z)
        h_hat = self._output_layer(z)

        return h_hat  # Channel estimate


class CGNN(nn.Module):
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
        **kwargs
    ):
        super().__init__()

        self._training = training
        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking
        self.dtype = dtype

        # Initialization for the state
        print("flag: CGNN")
        print(layer_type_dense)
        if self._var_mcs_masking:
            self._s_init = nn.ModuleList(
                [StateInit(d_s, num_units_init, layer_type=layer_type_conv)]
            )
        else:
            self._s_init = nn.ModuleList(
                [
                    StateInit(d_s, num_units_init, layer_type=layer_type_conv)
                    for _ in num_bits_per_symbol
                ]
            )

        # Iterations blocks
        self._iterations = nn.ModuleList(
            [
                CGNNIt(
                    d_s,
                    num_units_agg[i],
                    num_units_state[i],
                    layer_type_dense=layer_type_dense,
                    layer_type_conv=layer_type_conv,
                )
                for i in range(num_it)
            ]
        )
        self._num_it = num_it

        # Readouts
        if self._var_mcs_masking:
            self._readout_llrs = nn.ModuleList(
                [
                    ReadoutLLRs(
                        max(num_bits_per_symbol),
                        num_units_readout,
                        layer_type=layer_type_readout,
                    )
                ]
            )
        else:
            self._readout_llrs = nn.ModuleList(
                [
                    ReadoutLLRs(
                        num_bits, num_units_readout, layer_type=layer_type_readout
                    )
                    for num_bits in num_bits_per_symbol
                ]
            )

        self._readout_chest = ReadoutChEst(
            num_rx_ant, num_units_readout, layer_type=layer_type_readout
        )

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

    # ... (rest of the methods remain the same)

    def forward(self, inputs):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs

        # Normalization
        norm_scaling = torch.mean(torch.square(y), dim=(1, 2, 3), keepdim=True)
        norm_scaling = torch.where(
            norm_scaling != 0,
            1.0 / torch.sqrt(norm_scaling),
            torch.tensor(1.0, device=y.device, dtype=self.dtype),
        )
        y = y * norm_scaling
        norm_scaling = norm_scaling.unsqueeze(1)
        if h_hat is not None:
            h_hat = h_hat * norm_scaling

        # State initialization
        if self._var_mcs_masking:
            s = self._s_init[0]((y, pe, h_hat))
        else:
            s = self._s_init[0]((y, pe, h_hat)) * expand_to_rank(
                mcs_ue_mask[:, :, 0:1], 5, axis=-1
            )
            for idx in range(1, self._num_mcss_supported):
                s = s + self._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                    mcs_ue_mask[:, :, idx : idx + 1], 5, axis=-1
                )

        # Run receiver iterations
        llrs = []
        h_hats = []
        for i in range(self._num_it):
            it = self._iterations[i]
            s = it([s, pe, active_tx])

            if (self._training and self._apply_multiloss) or i == self._num_it - 1:
                llrs_ = []
                for idx in range(self._num_mcss_supported):
                    if self._var_mcs_masking:
                        llrs__ = self._readout_llrs[0](s)
                        llrs__ = llrs__[..., : self._num_bits_per_symbol[idx]]
                    else:
                        llrs__ = self._readout_llrs[idx](s)
                    llrs_.append(llrs__)
                llrs.append(llrs_)
                h_hats.append(self._readout_chest(s))

        return llrs, h_hats


##########################3
############################3
##############################
#######################
#################################
###################################
############################


class CGNNOFDM(nn.Module):
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
        layer_type_dense="linear",
        layer_type_conv="separable_conv2d",
        layer_type_readout="linear",
        nrx_dtype=torch.float32,
    ):
        super().__init__()
        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype
        self._num_mcss_supported = len(sys_parameters.mcs_index)
        self._rg = ResourceGridWrapper(sys_parameters.transmitters[0]._resource_grid)

        if self._sys_parameters.mask_pilots:
            print("Masking pilots for pilotless communications.")

        self._mcs_var_mcs_masking = False
        if hasattr(self._sys_parameters, "mcs_var_mcs_masking"):
            self._mcs_var_mcs_masking = self._sys_parameters.mcs_var_mcs_masking
            print("Var-MCS NRX with masking.")
        elif len(sys_parameters.mcs_index) > 1:
            print("Var-MCS NRX with MCS-specific IO layers.")
        else:
            # Single-MCS NRX.
            pass

        # all UEs in the same pusch config must use the same MCS
        num_bits_per_symbol = []
        for mcs_list_idx in range(self._num_mcss_supported):
            num_bits_per_symbol.append(
                sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol
            )

        # Number of receive antennas
        num_rx_ant = sys_parameters.num_rx_antennas

        # Core neural receiver
        self._cgnn = CGNN(
            num_bits_per_symbol,  # is a list
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

        # Resource grid demapper to extract the data-carrying resource elements from the resource grid
        self._rg_demapper = ResourceGridDemapper(self._rg, sys_parameters.sm)

        # Instantiate the loss function if training
        if training:
            self._bce = nn.BCEWithLogitsLoss(reduction="none")
            self._mse = nn.MSELoss(reduction="none")

        # Pre-compute positional encoding
        rg_type = self._rg.build_type_grid()[:, 0]  # One stream only
        pilot_ind = torch.where(rg_type == 1)

        # Convert TensorFlow tensor to PyTorch tensor
        pilots = torch.from_numpy(self._rg.pilot_pattern.pilots.numpy())
        pilots = flatten_last_dims(pilots, 3)

        pilots_only = torch.zeros_like(rg_type, dtype=torch.complex64)
        pilots_only[pilot_ind] = pilots

        pilot_ind = torch.nonzero(torch.abs(pilots_only) > 1e-3)
        pilot_ind_sorted = [[] for _ in range(max_num_tx)]
        for p_ind in pilot_ind:
            tx_ind = p_ind[0].item()
            re_ind = p_ind[1:].tolist()
            pilot_ind_sorted[tx_ind].append(re_ind)

        pilots_dist_time = torch.zeros(
            max_num_tx,
            self._rg.num_ofdm_symbols,
            self._rg.fft_size,
            len(pilot_ind_sorted[0]),
        )
        pilots_dist_freq = torch.zeros(
            max_num_tx,
            self._rg.num_ofdm_symbols,
            self._rg.fft_size,
            len(pilot_ind_sorted[0]),
        )

        t_ind = torch.arange(self._rg.num_ofdm_symbols)
        f_ind = torch.arange(self._rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):
                pt = torch.abs(p_ind[0] - t_ind).unsqueeze(1)
                pilots_dist_time[tx_ind, :, :, i] = pt
                pf = torch.abs(p_ind[1] - f_ind).unsqueeze(0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        nearest_pilot_dist_time = torch.min(pilots_dist_time, dim=-1).values
        nearest_pilot_dist_freq = torch.min(pilots_dist_freq, dim=-1).values

        nearest_pilot_dist_time -= torch.mean(
            nearest_pilot_dist_time, dim=1, keepdim=True
        )
        std_ = torch.std(nearest_pilot_dist_time, dim=1, keepdim=True)
        nearest_pilot_dist_time = torch.where(
            std_ > 0.0, nearest_pilot_dist_time / std_, nearest_pilot_dist_time
        )

        nearest_pilot_dist_freq -= torch.mean(
            nearest_pilot_dist_freq, dim=2, keepdim=True
        )
        std_ = torch.std(nearest_pilot_dist_freq, dim=2, keepdim=True)
        nearest_pilot_dist_freq = torch.where(
            std_ > 0.0, nearest_pilot_dist_freq / std_, nearest_pilot_dist_freq
        )

        nearest_pilot_dist = torch.stack(
            [nearest_pilot_dist_time, nearest_pilot_dist_freq], dim=-1
        )
        self._nearest_pilot_dist = nearest_pilot_dist.permute(0, 2, 1, 3)

    @property
    def num_it(self):
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
                mcs_ue_mask = torch.nn.functional.one_hot(
                    torch.tensor(mcs_arr_eval[0]), num_classes=self._num_mcss_supported
                )
            else:
                mcs_ue_mask = mcs_ue_mask_eval

        mcs_ue_mask = mcs_ue_mask.unsqueeze(0).unsqueeze(0)
        num_tx = active_tx.shape[1]

        if self._sys_parameters.mask_pilots:
            rg_type = torch.from_numpy(self._rg.build_type_grid().numpy())
            rg_type = rg_type.unsqueeze(0).expand(y.shape)
            y = torch.where(
                rg_type == 1, torch.tensor(0.0, dtype=y.dtype, device=y.device), y
            )

        y = y[:, 0]
        y = y.permute(0, 3, 2, 1)
        y = torch.cat([y.real, y.imag], dim=-1)

        pe = self._nearest_pilot_dist[:num_tx]

        y = y.to(self._nrx_dtype)
        pe = pe.to(self._nrx_dtype)
        if h_hat_init is not None:
            h_hat_init = h_hat_init.to(self._nrx_dtype)
        active_tx = active_tx.to(self._nrx_dtype)

        llrs_, h_hats_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask])

        indices = mcs_arr_eval
        llrs = []
        h_hats = []

        for llrs_, h_hat_ in zip(llrs_, h_hats_):
            h_hat_ = h_hat_.float()
            _llrs_ = []
            for idx in indices:
                llrs_[idx] = llrs_[idx].float()
                llrs_[idx] = llrs_[idx].permute(0, 1, 3, 2, 4)
                llrs_[idx] = llrs_[idx].unsqueeze(1)
                llrs_[idx] = self._rg_demapper(llrs_[idx])
                llrs_[idx] = llrs_[idx][:, :num_tx]
                llrs_[idx] = flatten_last_dims(llrs_[idx], 2)
                if self._layer_demappers is None:
                    llrs_[idx] = llrs_[idx].squeeze(-2)
                else:
                    llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                _llrs_.append(llrs_[idx])
            llrs.append(_llrs_)
            h_hats.append(h_hat_)

        if self._training:
            loss_data = torch.tensor(0.0, dtype=torch.float32, device=y.device)
            for llrs_ in llrs:
                for idx in range(len(indices)):
                    loss_data_ = self._bce(llrs_[idx], bits[idx])
                    mcs_ue_mask_ = torch.index_select(
                        mcs_ue_mask, 2, torch.tensor([indices[idx]], device=y.device)
                    )
                    mcs_ue_mask_ = mcs_ue_mask_.expand_as(loss_data_)
                    loss_data_ = torch.mul(loss_data_, mcs_ue_mask_)
                    active_tx_data = active_tx.unsqueeze(-1).expand_as(loss_data_)
                    loss_data_ = torch.mul(loss_data_, active_tx_data)
                    loss_data += torch.mean(loss_data_)

            loss_chest = torch.tensor(0.0, dtype=torch.float32, device=y.device)
            if h_hats is not None:
                for h_hat_ in h_hats:
                    if h is not None:
                        loss_chest += self._mse(h, h_hat_)
                active_tx_chest = active_tx.unsqueeze(-1).expand_as(loss_chest)
                loss_chest = torch.mul(loss_chest, active_tx_chest)
                loss_chest = torch.mean(loss_chest)

            return loss_data, loss_chest
        else:
            return llrs[-1][0], h_hats[-1]


###########################3
###########################3
###########################3
###########################3
###########################3


class TBEncoderWrapper(nn.Module):
    def __init__(self, tb_encoder):
        super().__init__()
        self.tb_encoder = tb_encoder

    def forward(self, x):
        return torch.from_numpy(self.tb_encoder(x.cpu().numpy()).numpy())


class TBDecoderWrapper(nn.Module):
    def __init__(self, tb_decoder):
        super().__init__()
        self.tb_decoder = tb_decoder

    def forward(self, x):
        output = self.tb_decoder(x.cpu().numpy())
        return tuple(torch.from_numpy(t.numpy()) for t in output)


class PUSCHLSChannelEstimatorWrapper(nn.Module):
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, inputs):
        y, no = inputs
        output = self.estimator([y.cpu().numpy(), no.cpu().numpy()])
        return tuple(torch.from_numpy(t.numpy()) for t in output)


class LayerDemapperWrapper(nn.Module):
    def __init__(self, layer_demapper):
        super().__init__()
        self.layer_demapper = layer_demapper

    def forward(self, x):
        return torch.from_numpy(self.layer_demapper(x.cpu().numpy()).numpy())


class NeuralPUSCHReceiver(nn.Module):
    def __init__(self, sys_parameters, training=False, **kwargs):
        super().__init__()

        self._sys_parameters = sys_parameters
        self._training = training

        # init transport block enc/decoder
        self._tb_encoders = nn.ModuleList()
        self._tb_decoders = nn.ModuleList()

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        for mcs_list_idx in range(self._num_mcss_supported):
            self._tb_encoders.append(
                TBEncoderWrapper(
                    self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder
                )
            )
            self._tb_decoders.append(
                TBDecoderWrapper(
                    TBDecoder(
                        self._tb_encoders[mcs_list_idx].tb_encoder,
                        num_bp_iter=sys_parameters.num_bp_iter,
                        cn_type=sys_parameters.cn_type,
                    )
                )
            )

        # Precoding matrix
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = torch.from_numpy(
                sys_parameters.transmitters[0]._precoder._w.numpy()
            )
        else:
            self._precoding_mat = torch.ones(
                sys_parameters.max_num_tx,
                sys_parameters.num_antenna_ports,
                1,
                dtype=torch.complex64,
            )

        # LS channel estimator
        rg = sys_parameters.transmitters[0]._resource_grid
        pc = sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimatorWrapper(
            PUSCHLSChannelEstimator(
                resource_grid=rg,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn",
            )
        )

        rg_type = torch.from_numpy(rg.build_type_grid().numpy())[:, 0]
        pilot_ind = torch.where(rg_type == 1)
        self._pilot_ind = pilot_ind[0].numpy()

        # Layer demappers
        self._layer_demappers = nn.ModuleList()
        for mcs_list_idx in range(self._num_mcss_supported):
            self._layer_demappers.append(
                LayerDemapperWrapper(
                    LayerDemapper(
                        self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                        sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol,
                    )
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
            # dtype=sys_parameters.nrx_dtype,
        )

    def estimate_channel(self, y, num_tx):
        if self._sys_parameters.initial_chest == "ls":
            if self._sys_parameters.mask_pilots:
                raise ValueError(
                    "Cannot use initial channel estimator if pilots are masked."
                )
            h_hat, _ = self._ls_est([y, torch.tensor(1e-1)])
            h_hat = h_hat[:, 0, :, :num_tx, 0]
            h_hat = h_hat.permute(0, 2, 4, 3, 1)
            h_hat = torch.cat([h_hat.real, h_hat.imag], dim=-1)
        elif self._sys_parameters.initial_chest is None:
            h_hat = None
        return h_hat

    def preprocess_channel_ground_truth(self, h):
        h = h.squeeze(1)
        h = h.permute(0, 2, 5, 4, 1, 3)
        w = self._precoding_mat.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = torch.matmul(h, w).squeeze(-1)
        h = torch.cat([h.real, h.imag], dim=-1)
        return h

    def forward(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        if self._training:
            y, active_tx, b, h, mcs_ue_mask = inputs
            if len(mcs_arr_eval) == 1 and not isinstance(b, list):
                b = [b]
            bits = []
            for idx in range(len(mcs_arr_eval)):
                bits.append(self._tb_encoders[mcs_arr_eval[idx]](b[idx]))

            num_tx = active_tx.shape[1]
            h_hat = self.estimate_channel(y, num_tx)

            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            losses = self._neural_rx(
                (y, h_hat, active_tx, bits, h, mcs_ue_mask), mcs_arr_eval
            )
            return losses
        else:
            y, active_tx = inputs
            num_tx = active_tx.shape[1]
            h_hat = self.estimate_channel(y, num_tx)

            llr, h_hat_refined = self._neural_rx(
                (y, h_hat, active_tx),
                [mcs_arr_eval[0]],
                mcs_ue_mask_eval=mcs_ue_mask_eval,
            )

            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)
            return b_hat, h_hat_refined, h_hat, tb_crc_status


################################
## ONNX Layers / Wrapper
################################
# The following layers provide an adapter to the Aerial PUSCH pipeline
# the code is only relevant for for ONNX/TensorRT exports but can be ignored
# for Sionna-based simulations.


class NRPreprocessing(nn.Module):
    def __init__(self, num_tx, **kwargs):
        super().__init__(**kwargs)
        self._num_tx = num_tx
        self._num_res_per_prb = 12  # fixed in 5G

    def _focc_removal(self, h_hat):
        """
        Apply FOCC removal to h_hat.
        """
        shape = [-1, 2]
        s = h_hat.shape
        new_shape = s[:3] + shape

        h_hat = h_hat.view(new_shape)
        h_hat = torch.sum(h_hat, dim=-1, keepdim=True) / 2.0
        h_hat = h_hat.repeat(1, 1, 1, 1, 2)
        h_ls = h_hat.view(s[:3] + (-1,))

        return h_ls

    def _calculate_nn_indices(
        self, dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
    ):
        """
        Calculates nearest neighbor interpolation indices for a single PRB.
        """
        re_pos = torch.stack(
            torch.meshgrid(
                torch.arange(self._num_res_per_prb), torch.arange(num_ofdm_symbols)
            ),
            dim=-1,
        )
        re_pos = re_pos.view(-1, 1, 2)

        pes = []
        nn_idxs = []

        for tx_idx in range(self._num_tx):
            pilot_pos = torch.stack(
                torch.meshgrid(dmrs_subcarrier_pos[tx_idx], dmrs_ofdm_pos[tx_idx]),
                dim=-1,
            )
            pilot_pos = pilot_pos.view(1, -1, 2)

            diff = torch.abs(re_pos - pilot_pos)
            dist = torch.sum(diff, dim=-1)
            nn_idx = torch.argmin(dist, dim=1)
            nn_idx = nn_idx.view(1, 1, num_ofdm_symbols, self._num_res_per_prb)

            pe = torch.min(diff, dim=1)[0]
            pe = pe.view(1, num_ofdm_symbols, self._num_res_per_prb, 2).permute(
                0, 2, 1, 3
            )

            pe = pe.float()
            p = []

            pe_ = pe[..., 1:2]
            pe_ -= pe_.mean()
            std_ = pe_.std()
            pe_ = torch.where(std_ > 0.0, pe_ / std_, pe_)
            p.append(pe_)

            pe_ = pe[..., 0:1]
            pe_ -= pe_.mean()
            std_ = pe_.std()
            pe_ = torch.where(std_ > 0.0, pe_ / std_, pe_)
            p.append(pe_)

            pe = torch.cat(p, dim=-1)
            pes.append(pe)
            nn_idxs.append(nn_idx)

        pe = torch.cat(pes, dim=0)
        pe = pe.repeat(1, num_prbs, 1, 1)
        nn_idx = torch.cat(nn_idxs, dim=0)

        return nn_idx, pe

    def _nn_interpolation(
        self, h_hat, num_ofdm_symbols, dmrs_ofdm_pos, dmrs_subcarrier_pos
    ):
        """
        Applies nearest neighbor interpolation of pilots to all data symbols in the resource grid.
        """
        num_pilots_per_dmrs = dmrs_subcarrier_pos.shape[1]
        num_prbs = h_hat.shape[-1] // (num_pilots_per_dmrs * dmrs_ofdm_pos.shape[-1])

        s = h_hat.shape
        h_hat = h_hat.view(*s[:3], -1, num_pilots_per_dmrs, num_prbs)
        h_hat = h_hat.permute(0, 1, 2, 4, 3, 5)
        h_hat = h_hat.reshape(s)

        h_hat = h_hat.unsqueeze(1).unsqueeze(4)
        h_hat = h_hat.permute(*torch.roll(torch.arange(h_hat.dim()), -3))

        ls_nn_ind, pe = self._calculate_nn_indices(
            dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
        )

        s = h_hat.shape
        h_hat_prb = h_hat.view(*s[:2], num_prbs, -1, *s[4:])
        h_hat_prb = h_hat_prb.permute(0, 1, 3, 2, 4, 5, 6)

        outputs = torch.gather(
            h_hat_prb,
            2,
            ls_nn_ind.expand(*h_hat_prb.shape[:2], -1, *h_hat_prb.shape[3:]),
        )
        outputs = outputs.permute(0, 1, 2, 4, 3, 5, 6, 7)

        s = outputs.shape
        s = s[:3] + (num_prbs * self._num_res_per_prb,) + s[5:]
        outputs = outputs.reshape(s)

        h_hat = outputs.permute(*torch.roll(torch.arange(outputs.dim()), 3))

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
    """
    Wraps the 5G NR neural receiver in an ONNX compatible format.

    Note that the shapes are optimized for Aerial and not directly compatible
    with Sionna.

    Parameters
    -----------
    num_it : int
        Number of iterations.
    d_s : int
        Size of the state vector.
    num_units_init : list of int
        Number of hidden units for the init network.
    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network
    num_units_state : list of int
        Number of hidden units for the state-update network.
    num_units_readout : list of int
        Number of hidden units for the read-out network.
    num_bits_per_symbol: int
        Number of bits per symbol.
    num_tx: int
        Max. number of layers/DMRS ports. Note that DMRS ports can be
        dynamically deactivated via the dmrs_port_mask.
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
        **kwargs
    ):
        super().__init__()
        assert len(num_units_agg) == num_it and len(num_units_state) == num_it

        # hard-coded for simplicity
        self._num_tx = num_tx  # we assume 1 stream per user

        ####################################################
        # Detector
        ####################################################
        self._cgnn = CGNN(
            [num_bits_per_symbol],  # no support for mixed MCS
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
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (
            val <= len(self._iterations)
        ), "Invalid number of iterations"
        self._num_it = val

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

        y = torch.cat((y_real, y_imag), dim=-1)
        h_hat_p = torch.cat((h_hat_real, h_hat_imag), dim=-1)

        # nearest neighbor interpolation of channel estimates
        h_hat, pe = self._preprocessing(
            (y, h_hat_p, dmrs_ofdm_pos, dmrs_subcarrier_pos)
        )

        # dummy MCS mask (no support for mixed MCS)
        mcs_ue_mask = torch.ones((1, 1, 1), dtype=torch.float32)

        # and run NRX
        llr, h_hat = self._cgnn([y, pe, h_hat, dmrs_port_mask, mcs_ue_mask])

        # cgnn returns list of results for each iteration
        # (not needed for inference)
        llr = llr[-1][0]  # take LLRs of first MCS (no support for mixed MCS)
        h_hat = h_hat[-1]

        # cast back to float32 (if NRX uses quantization)
        llr = llr.float()
        h_hat = h_hat.float()

        # reshape llrs in Aerial format
        llr = llr.permute(0, 4, 1, 2, 3)
        # Sionna defines LLRs with different sign
        llr = -1.0 * llr

        return llr, h_hat
