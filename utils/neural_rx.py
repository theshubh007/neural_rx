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

# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, Layer
# from tensorflow.nn import relu
from sionna.utils import (
    flatten_last_dims,
)
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator


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
        **kwargs
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
        **kwargs
    ):
        print("AggregateUserStates")
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Initialize hidden layers with both in_features and out_features
        self._hidden_layers = nn.ModuleList()
        print("flag0")
        for n in num_units:
            print("n: ", n)
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
    r"""
    Implements an iteration of the CGNN detector.

    Consists of two stages: State aggregation followed by state update.
    """

    def __init__(
        self,
        d_s,
        num_units_agg,
        num_units_state_update,
        layer_type_dense="dense",
        layer_type_conv="sepconv",
        dtype=torch.float32,
        **kwargs
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
    r"""
    Network computing LLRs from the state vectors.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.
    """

    def __init__(
        self,
        num_bits_per_symbol,
        num_units,
        layer_type="dense",
        dtype=torch.float32,
        **kwargs
    ):
        super().__init__()

        # Choose the correct layer type
        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden layers
        self._hidden_layers = nn.ModuleList([layer(n, dtype=dtype) for n in num_units])
        self._output_layer = layer(num_bits_per_symbol, dtype=dtype)

    def forward(self, s):
        # Input of the MLP
        z = s
        # Apply MLP
        for layer in self._hidden_layers:
            z = F.relu(layer(z))
        llr = self._output_layer(z)

        return llr  # LLRs on the transmitted bits


class ReadoutChEst(nn.Module):
    r"""
    Network computing channel estimate.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `2*num_rx_ant` units.
    """

    def __init__(
        self, num_rx_ant, num_units, layer_type="dense", dtype=torch.float32, **kwargs
    ):
        super().__init__()

        if layer_type == "dense":
            layer = nn.Linear
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden layers
        self._hidden_layers = nn.ModuleList([layer(n, dtype=dtype) for n in num_units])
        self._output_layer = layer(2 * num_rx_ant, dtype=dtype)

    def forward(self, s):
        # Input of the MLP
        z = s
        # Apply MLP
        for layer in self._hidden_layers:
            z = F.relu(layer(z))
        h_hat = self._output_layer(z)

        return h_hat  # Channel estimate


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
        **kwargs
    ):
        super().__init__()

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
        if self._var_mcs_masking:
            self._readout_llrs = [
                ReadoutLLRs(
                    max(num_bits_per_symbol),
                    num_units_readout,
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
                        layer_type=layer_type_readout,
                        dtype=dtype,
                    )
                )

        # Channel estimate readout
        self._readout_chest = ReadoutChEst(
            num_rx_ant, num_units_readout, layer_type=layer_type_readout, dtype=dtype
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
        **kwargs
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
            pass  # Single-MCS NRX

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
        ##############################################
        rg_type = self._rg.build_type_grid()[:, 0]  # One stream only
        pilot_ind = torch.where(rg_type == 1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        pilots_only = torch.zeros(rg_type.shape).scatter_(0, pilot_ind, pilots)
        pilot_ind = torch.where(torch.abs(pilots_only) > 1e-3)

        # Sort the pilots according to which TX they are allocated
        pilot_ind_sorted = [[] for _ in range(max_num_tx)]
        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = torch.tensor(pilot_ind_sorted)

        # Distance to the nearest pilot in time and frequency
        pilots_dist_time = torch.zeros(
            (
                max_num_tx,
                self._rg.num_ofdm_symbols,
                self._rg.fft_size,
                pilot_ind_sorted.shape[1],
            )
        )
        pilots_dist_freq = torch.zeros(
            (
                max_num_tx,
                self._rg.num_ofdm_symbols,
                self._rg.fft_size,
                pilot_ind_sorted.shape[1],
            )
        )

        t_ind = torch.arange(self._rg.num_ofdm_symbols)
        f_ind = torch.arange(self._rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):
                pilots_dist_time[tx_ind, :, :, i] = torch.abs(
                    p_ind[0] - t_ind
                ).unsqueeze(1)
                pilots_dist_freq[tx_ind, :, :, i] = torch.abs(
                    p_ind[1] - f_ind
                ).unsqueeze(0)

        # Normalize the distances
        nearest_pilot_dist_time = pilots_dist_time.min(dim=-1)[0]
        nearest_pilot_dist_time -= nearest_pilot_dist_time.mean(dim=1, keepdim=True)
        std_ = nearest_pilot_dist_time.std(dim=1, keepdim=True)
        nearest_pilot_dist_time = torch.where(
            std_ > 0.0, nearest_pilot_dist_time / std_, nearest_pilot_dist_time
        )
        nearest_pilot_dist_freq = pilots_dist_freq.min(dim=-1)[0]
        nearest_pilot_dist_freq -= nearest_pilot_dist_freq.mean(dim=2, keepdim=True)
        std_ = nearest_pilot_dist_freq.std(dim=2, keepdim=True)
        nearest_pilot_dist_freq = torch.where(
            std_ > 0.0, nearest_pilot_dist_freq / std_, nearest_pilot_dist_freq
        )

        nearest_pilot_dist = torch.stack(
            [nearest_pilot_dist_time, nearest_pilot_dist_freq], dim=-1
        )
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


class NeuralPUSCHReceiver(nn.Module):
    def __init__(self, sys_parameters, training=False, **kwargs):
        super().__init__()

        self._sys_parameters = sys_parameters
        self._training = training

        # init transport block encoder/decoder
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

        # Precoding matrix to post-process the ground-truth channel when training
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = torch.ones(
                [sys_parameters.max_num_tx, sys_parameters.num_antenna_ports, 1],
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
        w = self._precoding_mat.unsqueeze(0).expand_as(h)
        h = (h @ w).squeeze(-1)
        h = torch.cat([h.real, h.imag], dim=-1)
        return h

    def forward(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        if self._training:
            y, active_tx, b, h, mcs_ue_mask = inputs
            if len(mcs_arr_eval) == 1 and not isinstance(b, list):
                b = [b]
            bits = [
                self._tb_encoders[mcs_arr_eval[idx]](b[idx])
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
            *outputs.shape[5:]
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
        **kwargs
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
