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
import numpy as np

# from tensorflow.keras import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

# from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, Layer
# from tensorflow.nn import relu
from sionna.utils import (
    flatten_dims,
    split_dim,
    flatten_last_dims,
    insert_dims,
    expand_to_rank,
)
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


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
    # pylint: disable=line-too-long
    r"""
    Network initializing the state tensor for each user.

    The network consist of len(num_units) hidden blocks, each block
    consisting of
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity

    Parameters
    -----------
    d_s : int
        Size of the state vector

    num_units : list of int
        Number of kernels for the hidden layers of the MLP.

    layer_type: str | "sepconv" | "conv"
        Defines which Convolutional layers are used. Will be either
        SeparableConv2D or Conv2D.

    Input
    ------
    (y, pe, h_hat)
    Tuple:

    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float
        The received OFDM resource grid after cyclic prefix removal and FFT.

    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency.
        This can be seen as a form of positional encoding.

    h_hat : None or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                     2*num_rx_ant], tf.float
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Initial state tensor for each user.
    """

    def __init__(
        self, d_s, num_units, layer_type="sepconv", dtype=torch.float32, **kwargs
    ):
        super().__init__(**kwargs)
        print("Init: StateInit")

        # allows for the configuration of multiple layer types
        # one could add custom layers here
        if layer_type == "sepconv":
            layer = SeparableConv2d
        # elif layer_type == "conv":
        # layer = Conv2d
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = nn.ModuleList()
        print("flag1")
        in_channels = 3  # Assuming input has 3 channels (y, pe, h_hat)
        for n in num_units:

            conv = nn.Sequential(layer(in_channels, n, kernel_size=3), nn.ReLU())
            print("flag2")
            self._hidden_conv.append(conv)
            in_channels = n

        #  Output block
        print("flag3")
        self._output_conv = layer(in_channels, d_s, kernel_size=3)

    def forward(self, inputs):
        y, pe, h_hat = inputs

        batch_size = y.shape[0]
        num_tx = pe.shape[0]

        y = y.unsqueeze(1).repeat(1, num_tx, 1, 1, 1)
        y = flatten_dims(y, 2, 0)

        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)

        if h_hat is not None:
            h_hat = flatten_dims(h_hat, 2, 0)
            z = torch.cat([y, pe, h_hat], dim=-1)
        else:
            z = torch.cat([y, pe], dim=-1)

        for conv in self._hidden_conv:
            z = conv(z)
        z = self._output_conv(z)

        s0 = split_dim(z, [batch_size, num_tx], 0)

        return s0


class AggregateUserStates(nn.Module):
    def __init__(self, d_s, num_units, layer_type="dense", dtype=torch.float32):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self.hidden_layers = nn.ModuleList()
        for n in num_units:
            self.hidden_layers.append(nn.Sequential(layer(d_s, n), nn.ReLU()))
        self.output_layer = layer(n, d_s)

    def forward(self, inputs):
        s, active_tx = inputs

        # Process s
        sp = s
        for layer in self.hidden_layers:
            sp = layer(sp)
        sp = self.output_layer(sp)

        # Aggregate all states
        active_tx = expand_to_rank(active_tx, sp.dim(), dim=-1)
        sp = torch.mul(sp, active_tx)

        # aggregate and remove self-state
        a = torch.sum(sp, dim=1, keepdim=True) - sp

        # scale by number of active users
        p = torch.sum(active_tx, dim=1, keepdim=True) - 1.0
        p = torch.relu(p)  # clip negative values to ignore non-active user

        # avoid 0 for single active user
        p = torch.where(p == 0.0, torch.ones_like(p), 1.0 / p.clamp(min=1e-10))

        # and scale states by number of aggregated users
        a = torch.mul(a, p)

        return a


class UpdateState(nn.Module):
    def __init__(self, d_s, num_units, layer_type="sepconv", dtype=torch.float32):
        super().__init__()

        if layer_type == "sepconv":
            layer = SeparableConv2d
        elif layer_type == "conv":
            layer = nn.Conv2d
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = nn.ModuleList()
        for n in num_units:
            conv = nn.Sequential(
                layer(
                    in_channels=None,
                    out_channels=n,
                    kernel_size=(3, 3),
                ),
                nn.ReLU(),
            )
            self._hidden_conv.append(conv)

        # Output block
        self._output_conv = layer(
            in_channels=None, out_channels=d_s, kernel_size=3, padding=1
        )

    def forward(self, inputs):
        s, a, pe = inputs

        batch_size = s.shape[0]
        num_tx = s.shape[1]

        # Stack the inputs
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)
        s = flatten_dims(s, 2, 0)
        a = flatten_dims(a, 2, 0)
        z = torch.cat([a, s, pe], dim=-1)

        # Apply the neural network
        for conv in self._hidden_conv:
            z = conv(z)
        z = self._output_conv(z)

        # Skip connection
        z = z + s

        # Unflatten
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new


class CGNNIt(nn.Module):
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
    def __init__(
        self, num_bits_per_symbol, num_units, layer_type="dense", dtype=torch.float32
    ):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self.hidden_layers = nn.ModuleList()
        for n in num_units:
            self.hidden_layers.append(nn.Sequential(layer(None, n), nn.ReLU()))
        self.output_layer = layer(None, num_bits_per_symbol)

    def forward(self, s):
        z = s
        for layer in self.hidden_layers:
            z = layer(z)
        llr = self.output_layer(z)
        return llr


class ReadoutChEst(nn.Module):
    def __init__(self, num_rx_ant, num_units, layer_type="dense", dtype=torch.float32):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self.hidden_layers = nn.ModuleList()
        for n in num_units:
            self.hidden_layers.append(nn.Sequential(layer(None, n), nn.ReLU()))
        self.output_layer = layer(None, 2 * num_rx_ant)

    def forward(self, s):
        z = s
        for layer in self.hidden_layers:
            z = layer(z)
        h_hat = self.output_layer(z)
        return h_hat


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
    ):
        super().__init__()

        self._training = training
        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking

        # Initialization for the state
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
                        num_bits,
                        num_units_readout,
                        layer_type=layer_type_readout,
                    )
                    for num_bits in num_bits_per_symbol
                ]
            )

        self._readout_chest = ReadoutChEst(
            num_rx_ant, num_units_readout, layer_type=layer_type_readout
        )

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

    @property
    def apply_multiloss(self):
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

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
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs

        # Normalization
        norm_scaling = torch.mean(y.pow(2), dim=(1, 2, 3), keepdim=True)
        norm_scaling = torch.reciprocal(torch.sqrt(norm_scaling)).clamp(min=1e-10)
        y = y * norm_scaling
        norm_scaling = norm_scaling.unsqueeze(1)
        if h_hat is not None:
            h_hat = h_hat * norm_scaling

        # State initialization
        if self._var_mcs_masking:
            s = self._s_init[0]((y, pe, h_hat))
        else:
            s = sum(
                self._s_init[idx]((y, pe, h_hat))
                * expand_to_rank(mcs_ue_mask[:, :, idx : idx + 1], 5, dim=-1)
                for idx in range(self._num_mcss_supported)
            )

        # Run receiver iterations
        llrs = []
        h_hats = []
        for i in range(self._num_it):
            s = self._iterations[i]([s, pe, active_tx])

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
        layer_type_dense="dense",
        layer_type_conv="sepconv",
        layer_type_readout="dense",
        nrx_dtype=torch.float32,
    ):
        super().__init__()

        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype

        self._num_mcss_supported = len(sys_parameters.mcs_index)

        self._rg = sys_parameters.transmitters[0]._resource_grid

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

        ####################################################
        # Core neural receiver
        ####################################################
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

        ###################################################
        # Resource grid demapper to extract the
        # data-carrying resource elements from the
        # resource grid
        ###################################################
        self._rg_demapper = ResourceGridDemapper(self._rg, sys_parameters.sm)

        #################################################
        # Instantiate the loss function if training
        #################################################
        if training:
            # Loss function
            self._bce = nn.BCEWithLogitsLoss(reduction="none")
            # Loss function
            self._mse = nn.MSELoss(reduction="none")

        ###############################################
        # Pre-compute positional encoding.
        # Positional encoding consists in the distance
        # to the nearest pilot in time and frequency.
        # It is therefore a 2D positional encoding.
        ##############################################

        # Indices of the pilot-carrying resource elements and pilot symbols
        rg_type = self._rg.build_type_grid()[:, 0]  # One stream only
        pilot_ind = torch.where(rg_type == 1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        # Resource grid carrying only the pilots
        # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
        pilots_only = torch.zeros_like(rg_type)
        pilots_only[pilot_ind] = pilots
        # Indices of pilots carrying RE (transmitter, freq, time)
        pilot_ind = torch.where(torch.abs(pilots_only) > 1e-3)
        pilot_ind = to_numpy(pilot_ind)

        # Sort the pilots according to which to which TX they are allocated
        pilot_ind_sorted = [[] for _ in range(max_num_tx)]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        # Distance to the nearest pilot in time
        # Initialized with zeros and then filled.
        pilots_dist_time = np.zeros(
            [
                max_num_tx,
                self._rg.num_ofdm_symbols,
                self._rg.fft_size,
                pilot_ind_sorted.shape[1],
            ]
        )
        # Distance to the nearest pilot in frequency
        # Initialized with zeros and then filled
        pilots_dist_freq = np.zeros(
            [
                max_num_tx,
                self._rg.num_ofdm_symbols,
                self._rg.fft_size,
                pilot_ind_sorted.shape[1],
            ]
        )

        t_ind = np.arange(self._rg.num_ofdm_symbols)
        f_ind = np.arange(self._rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):

                pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
                pilots_dist_time[tx_ind, :, :, i] = pt

                pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        # Normalizing the tensors of distance to force zero-mean and
        # unit variance.
        nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
        nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
        nearest_pilot_dist_time -= np.mean(
            nearest_pilot_dist_time, axis=1, keepdims=True
        )
        std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
        nearest_pilot_dist_time = np.where(
            std_ > 0.0, nearest_pilot_dist_time / std_, nearest_pilot_dist_time
        )
        nearest_pilot_dist_freq -= np.mean(
            nearest_pilot_dist_freq, axis=2, keepdims=True
        )
        std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
        nearest_pilot_dist_freq = np.where(
            std_ > 0.0, nearest_pilot_dist_freq / std_, nearest_pilot_dist_freq
        )

        # Stacking the time and frequency distances and casting to PyTorch types.
        nearest_pilot_dist = np.stack(
            [nearest_pilot_dist_time, nearest_pilot_dist_freq], axis=-1
        )
        nearest_pilot_dist = torch.tensor(nearest_pilot_dist, dtype=torch.float32)
        # Reshaping to match the expected shape.
        # [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
        self._nearest_pilot_dist = nearest_pilot_dist.permute(0, 2, 1, 3)

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
                )
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, dim=0)

        num_tx = active_tx.shape[1]

        if self._sys_parameters.mask_pilots:
            rg_type = self._rg.build_type_grid()
            rg_type = rg_type.unsqueeze(0)
            rg_type = rg_type.expand(y.shape)
            y = torch.where(rg_type == 1, torch.zeros_like(y), y)

        y = y[:, 0]
        y = y.permute(0, 3, 2, 1)
        y = torch.cat([y.real, y.imag], dim=-1)

        pe = self._nearest_pilot_dist[:num_tx]

        y = y.to(dtype=self._nrx_dtype)
        pe = pe.to(dtype=self._nrx_dtype)

        if h_hat_init is not None:
            h_hat_init = h_hat_init.to(dtype=self._nrx_dtype)
        active_tx = active_tx.to(dtype=self._nrx_dtype)

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
            loss_data = torch.tensor(0.0, dtype=torch.float32)
            for llrs_ in llrs:
                for idx, llr in enumerate(llrs_):
                    loss_data_ = self._bce(llr, bits[idx])
                    mcs_ue_mask_ = expand_to_rank(
                        mcs_ue_mask[:, :, indices[idx]], loss_data_.dim(), dim=-1
                    )
                    loss_data_ = loss_data_ * mcs_ue_mask_
                    active_tx_data = expand_to_rank(active_tx, loss_data_.dim(), dim=-1)
                    loss_data_ = loss_data_ * active_tx_data
                    loss_data += loss_data_.mean()

            loss_chest = torch.tensor(0.0, dtype=torch.float32)
            if h_hats is not None and h is not None:
                for h_hat_ in h_hats:
                    loss_chest += self._mse(h, h_hat_)

            active_tx_chest = expand_to_rank(active_tx, loss_chest.dim(), dim=-1)
            loss_chest = loss_chest * active_tx_chest
            loss_chest = loss_chest.mean()

            return loss_data, loss_chest
        else:
            return llrs[-1][0], h_hats[-1]


class LayerDemapper(nn.Module):
    def __init__(self, layer_mapper, num_bits_per_symbol):
        super().__init__()
        self.layer_mapper = layer_mapper
        self.num_bits_per_symbol = num_bits_per_symbol

    def forward(self, x):
        # Implement the forward pass logic here
        # This will depend on what the original LayerDemapper does
        # You may need to port the TensorFlow implementation to PyTorch
        pass


class NeuralPUSCHReceiver(nn.Module):
    def __init__(self, sys_parameters, training=False):
        super().__init__()

        self._sys_parameters = sys_parameters
        self._training = training

        # init transport block enc/decoder
        self._tb_encoders = []
        self._tb_decoders = []

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

        # Precoding matrix
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = torch.ones(
                (sys_parameters.max_num_tx, sys_parameters.num_antenna_ports, 1),
                dtype=torch.complex64,
            )

        # LS channel estimator
        rg = sys_parameters.transmitters[0]._resource_grid
        pc = sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimator(
            resource_grid=rg,
            dmrs_length=pc.dmrs.length,
            dmrs_additional_position=pc.dmrs.additional_position,
            num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
            interpolation_type="nn",
        )

        rg_type = rg.build_type_grid()[:, 0].numpy()
        rg_type = torch.from_numpy(rg_type)
        pilot_ind = torch.where(rg_type == 1)
        self._pilot_ind = np.array(pilot_ind)

        # # Layer demappers
        # self._layer_demappers = nn.ModuleList(
        #     [
        #         LayerDemapper(
        #             self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
        #             sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol,
        #         )
        #         for mcs_list_idx in range(self._num_mcss_supported)
        #     ]
        # )
        self._layer_demappers = nn.ModuleList(
            [
                LayerDemapper(
                    self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                    sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol,
                )
                for mcs_list_idx in range(self._num_mcss_supported)
            ]
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
            # dtype=torch.float32,
        )

    def estimate_channel(self, y, num_tx):
        if self._sys_parameters.initial_chest == "ls":
            if self._sys_parameters.mask_pilots:
                raise ValueError(
                    "Cannot use initial channel estimator if pilots are masked."
                )
            h_hat, _ = self._ls_est([y, 1e-1])
            h_hat = h_hat[:, 0, :, :num_tx, 0]
            h_hat = h_hat.permute(0, 2, 4, 3, 1)
            h_hat = torch.cat([h_hat.real, h_hat.imag], dim=-1)
        elif self._sys_parameters.initial_chest is None:
            h_hat = None
        return h_hat

    def preprocess_channel_ground_truth(self, h):
        h = h.squeeze(1)
        h = h.permute(0, 2, 5, 4, 1, 3)
        w = insert_dims(self._precoding_mat.unsqueeze(0), 2, 2)
        h = torch.matmul(h, w).squeeze(-1)
        h = torch.cat([h.real, h.imag], dim=-1)
        return h

    def forward(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        if self._training:
            y, active_tx, b, h, mcs_ue_mask = inputs
            if len(mcs_arr_eval) == 1 and not isinstance(b, list):
                b = [b]
            bits = [
                self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx])
                for idx in range(len(mcs_arr_eval))
            ]
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
    def __init__(self, num_tx):
        super().__init__()
        self._num_tx = num_tx
        self._num_res_per_prb = 12  # fixed in 5G

    def _focc_removal(self, h_hat):
        shape = [-1, 2]
        s = h_hat.shape
        new_shape = s[:3] + shape
        h_hat = h_hat.reshape(new_shape)

        h_hat = torch.sum(h_hat, dim=-1, keepdim=True) / 2.0
        h_hat = h_hat.repeat(1, 1, 1, 1, 2)

        shape = [-1]
        s = h_hat.shape
        new_shape = s[:3] + shape
        h_ls = h_hat.reshape(new_shape)

        return h_ls

    def _calculate_nn_indices(
        self, dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
    ):
        re_pos = torch.stack(
            torch.meshgrid(
                torch.arange(self._num_res_per_prb), torch.arange(num_ofdm_symbols)
            ),
            dim=-1,
        )
        re_pos = re_pos.reshape(-1, 1, 2)

        pes = []
        nn_idxs = []
        for tx_idx in range(self._num_tx):
            p_idx = torch.stack(
                torch.meshgrid(dmrs_subcarrier_pos[tx_idx], dmrs_ofdm_pos[tx_idx]),
                dim=-1,
            )
            pilot_pos = p_idx.reshape(-1, 2)
            pilot_pos = pilot_pos.unsqueeze(0)
            diff = torch.abs(re_pos - pilot_pos)
            dist = torch.sum(diff, dim=-1)

            nn_idx = torch.argmin(dist, dim=1)
            nn_idx = nn_idx.reshape(1, 1, num_ofdm_symbols, self._num_res_per_prb)

            pe = torch.min(diff, dim=1)[0]
            pe = pe.reshape(1, num_ofdm_symbols, self._num_res_per_prb, 2)
            pe = pe.permute(0, 2, 1, 3)

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
        nn_idx = torch.cat(nn_idxs, dim=0)
        return nn_idx, pe

    def _nn_interpolation(
        self, h_hat, num_ofdm_symbols, dmrs_ofdm_pos, dmrs_subcarrier_pos
    ):
        num_pilots_per_dmrs = dmrs_subcarrier_pos.shape[1]
        num_prbs = int(
            h_hat.shape[-1] / (num_pilots_per_dmrs * dmrs_ofdm_pos.shape[-1])
        )

        s = h_hat.shape
        h_hat = split_dim(h_hat, shape=(-1, num_pilots_per_dmrs), dim=3)
        h_hat = split_dim(h_hat, shape=(-1, num_prbs), dim=3)
        h_hat = h_hat.permute(0, 1, 2, 4, 3, 5)
        h_hat = h_hat.reshape(s)

        h_hat = h_hat.unsqueeze(1).unsqueeze(4)
        perm = torch.roll(torch.arange(h_hat.dim()), -3, 0)
        h_hat = h_hat.permute(*perm)

        ls_nn_ind, pe = self._calculate_nn_indices(
            dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
        )

        s = h_hat.shape
        h_hat_prb = split_dim(h_hat, shape=(num_prbs, -1), dim=2)
        h_hat_prb = h_hat_prb.permute(0, 1, 3, 2, 4, 5, 6)
        outputs = torch.gather(
            h_hat_prb,
            2,
            ls_nn_ind.expand(*h_hat_prb.shape[:2], -1, *h_hat_prb.shape[3:]),
        )
        outputs = outputs.permute(0, 1, 2, 4, 3, 5, 6, 7)

        s = outputs.shape
        s = torch.Size((-1, *s[1:3], num_prbs * self._num_res_per_prb, *s[5:]))
        outputs = outputs.reshape(s)

        perm = torch.roll(torch.arange(outputs.dim()), 3, 0)
        h_hat = outputs.permute(*perm)
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
    ):
        super().__init__()
        assert len(num_units_agg) == num_it and len(num_units_state) == num_it

        self._num_tx = num_tx  # we assume 1 stream per user

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
            dtype=getattr(torch, nrx_dtype.name),
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
