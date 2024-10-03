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
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import Model
from sionna.utils import (
    flatten_dims,
    split_dim,
    # expand_to_rank,
)
import numpy as np
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, PUSCHLSChannelEstimator
from sionna.nr import LayerDemapper as SionnaLayerDemapper


def ensure_torch_tensor(tensor):
    if isinstance(tensor, tf.Tensor):
        return torch.from_numpy(tensor.numpy())
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return torch.tensor(tensor)


def expand_to_rank(tensor, rank, axis=0):
    if isinstance(tensor, (int, float)):
        tensor = torch.tensor([tensor])
    elif isinstance(tensor, tf.Tensor) and tensor.shape == ():
        tensor = torch.tensor([tensor.numpy().item()])
    elif isinstance(tensor, torch.Tensor) and tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)

    while len(tensor.shape) < rank:
        if axis == -1:
            tensor = tensor.unsqueeze(-1)
        else:
            tensor = tensor.unsqueeze(0)
    return tensor


class StateInit(nn.Module):
    """
    Network initializing the state tensor for each user.

    The network consists of len(num_units) hidden blocks, each block
    consisting of:
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation
    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity

    Parameters
    ----------
    d_s : int
        Size of the state vector
    num_units : list of int
        Number of kernels for the hidden layers of the MLP.
    layer_type: str
        Defines which Convolutional layers are used. Can be "sepconv" or "conv".

    Input
    -----
    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], torch.Tensor
        The received OFDM resource grid after cyclic prefix removal and FFT.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.
    h_hat : None or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], torch.Tensor
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    Output
    ------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Initial state tensor for each user.
    """

    def __init__(self, d_s, num_units, layer_type="sepconv", dtype=torch.float32):
        super().__init__()

        if layer_type == "sepconv":

            def conv_layer(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=1,
                        groups=in_channels,
                    ),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                )

        elif layer_type == "conv":

            def conv_layer(in_channels, out_channels):
                return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        else:
            raise NotImplementedError(f"Unknown layer_type '{layer_type}' selected.")

        # Hidden blocks
        self.hidden_conv = nn.ModuleList()
        for n in num_units:
            self.hidden_conv.append(nn.Sequential(conv_layer(n, n), nn.ReLU()))

        # Output block
        self.output_conv = conv_layer(num_units[-1], d_s)

        # Convert all layers to specified dtype
        self = self.to(dtype)

    def forward(self, inputs):
        y, pe, h_hat = inputs

        # Get the input shapes dynamically
        batch_size = y.shape[0]
        num_tx = 2  # Assuming fixed number of transmitters (adjust if needed)
        num_subcarriers = y.shape[1]
        num_ofdm_symbols = y.shape[2]

        # Print shapes for debugging purposes (optional)
        print(
            f"Input shapes: y: {y.shape}, pe: {pe.shape}, h_hat: {h_hat.shape if h_hat is not None else 'None'}"
        )

        # Dynamically compute the reshape size based on the total number of elements in pe
        pe_num_elements = pe.numel()  # Total number of elements in pe tensor
        expected_elements = batch_size * num_tx

        # If the total number of elements in pe is less than expected, try to broadcast or pad it
        if pe_num_elements < expected_elements:
            print(
                f"Warning: pe_num_elements ({pe_num_elements}) is smaller than expected ({expected_elements}). Adjusting dynamically."
            )
            # Pad pe to match the expected size
            padding = expected_elements - pe_num_elements
            pe = torch.nn.functional.pad(pe, (0, padding))  # Pad on the last dimension
            pe_num_elements = pe.numel()

        # If more elements are present, reshape appropriately
        remaining_dim = pe_num_elements // expected_elements

        # Reshape pe
        if pe_num_elements % expected_elements != 0:
            raise ValueError(
                f"Cannot reshape pe of size {pe_num_elements} into batch_size: {batch_size}, num_tx: {num_tx}"
            )

        pe = pe.view(batch_size, num_tx, remaining_dim)

        if h_hat is not None:
            # Dynamically reshape h_hat as well
            h_hat_num_elements = h_hat.numel()
            remaining_h_hat_dim = h_hat_num_elements // expected_elements

            if h_hat_num_elements % expected_elements != 0:
                raise ValueError(
                    f"Cannot reshape h_hat of size {h_hat_num_elements} into batch_size: {batch_size}, num_tx: {num_tx}"
                )

            h_hat = h_hat.view(batch_size, num_tx, remaining_h_hat_dim)
            z = torch.cat([y.unsqueeze(1).expand(-1, num_tx, -1), pe, h_hat], dim=-1)
        else:
            z = torch.cat([y.unsqueeze(1).expand(-1, num_tx, -1), pe], dim=-1)

        # Apply the neural network
        z = z.view(batch_size * num_tx, -1, 1, 1)
        for conv in self.hidden_conv:
            z = conv(z)
        z = self.output_conv(z)

        # Reshape output back to the original format
        z = z.view(batch_size, num_tx, num_subcarriers, num_ofdm_symbols, -1)

        return z  # Initial state of every user


class AggregateUserStates(nn.Module):
    """
    For every user n, aggregate the states of all the other users n'!= n.
    An MLP is applied to every state before aggregating.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a linear layer without non-linearity and with `d_s` units.

    The input `active_tx` provides a mask of active users and non-active users
    will be ignored in the aggregation.

    Parameters
    ----------
    d_s : int
        Size of the state vector
    num_units : list of int
        Number of units for the hidden layers.
    layer_type: str
        Defines which Dense layers are used. Currently only supports "linear".

    Input
    -----
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        State tensor.
    active_tx: [batch_size, num_tx], torch.Tensor
        Active user mask.

    Output
    ------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        For every user `n`, aggregate state of the other users, i.e.,
        sum(s, axis=1) - s[:,n,:,:,:]
    """

    def __init__(self, d_s, num_units, layer_type="linear", dtype=torch.float32):
        super().__init__()

        if layer_type == "linear":
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(d_s, n, dtype=dtype) for n in num_units]
            )
            self.output_layer = nn.Linear(num_units[-1], d_s, dtype=dtype)
        elif layer_type == "conv1d":
            self.hidden_layers = nn.ModuleList(
                [nn.Conv1d(d_s, n, kernel_size=1, dtype=dtype) for n in num_units]
            )
            self.output_layer = nn.Conv1d(
                num_units[-1], d_s, kernel_size=1, dtype=dtype
            )
        else:
            raise NotImplementedError(
                f"Layer type '{layer_type}' is not supported. Use 'linear' or 'conv1d'."
            )

        self.activation = nn.ReLU()
        self.layer_type = layer_type

    def forward(self, inputs):
        s, active_tx = inputs

        # Process s
        sp = s
        if self.layer_type == "conv1d":
            # For Conv1d, we need to change the input shape
            sp = sp.transpose(1, 2)  # Change from [B, C, ...] to [B, ..., C]

        for layer in self.hidden_layers:
            sp = self.activation(layer(sp))
        sp = self.output_layer(sp)

        if self.layer_type == "conv1d":
            # Change back to original shape
            sp = sp.transpose(1, 2)

        # Aggregate all states
        active_tx = expand_to_rank(active_tx, sp.dim(), axis=-1)
        sp = torch.mul(sp, active_tx)

        # Aggregate and remove self-state
        a = torch.sum(sp, dim=1, keepdim=True) - sp

        # Scale by number of active users
        p = torch.sum(active_tx, dim=1, keepdim=True) - 1.0
        p = torch.clamp(p, min=0.0)  # clip negative values to ignore non-active user
        p = torch.where(p == 0.0, torch.ones_like(p), 1.0 / p)

        # Scale states by number of aggregated users
        a = torch.mul(a, p)

        return a


class UpdateState(nn.Module):
    """
    Updates the state tensor.

    The network consists of len(num_units) hidden blocks, each block consisting of:
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation
    The last block is the output block and has the same architecture, but with `d_s` units and no non-linearity.
    The network ends with a skip connection with the state.

    Parameters
    ----------
    d_s : int
        Size of the state vector.
    num_units : list of int
        Number of kernels for the hidden separable convolutional layers.
    layer_type: str
        Defines which Convolutional layers are used. Can be "sepconv" or "conv".

    Input
    -----
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        State tensor.
    a : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Aggregated states from other users.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time and frequency.
        This can be seen as a form of positional encoding.

    Output
    ------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Updated channel state vector.
    """

    def __init__(self, d_s, num_units, layer_type="sepconv"):
        super().__init__()

        if layer_type == "sepconv":
            conv_layer = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1),
                nn.Conv2d(1, 1, kernel_size=1),
            )
        elif layer_type == "conv":
            conv_layer = nn.Conv2d
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self.hidden_conv = nn.ModuleList()
        for n in num_units:
            self.hidden_conv.append(
                nn.Sequential(conv_layer(n, n, kernel_size=3, padding=1), nn.ReLU())
            )

        # Output block
        self.output_conv = conv_layer(num_units[-1], d_s, kernel_size=3, padding=1)

    def forward(self, inputs):
        s, a, pe = inputs

        batch_size = s.shape[0]
        num_tx = s.shape[1]

        # Stack the inputs
        pe = pe.repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)
        s = flatten_dims(s, 2, 0)
        a = flatten_dims(a, 2, 0)

        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 2*d_s + 2]
        z = torch.cat([a, s, pe], dim=-1)

        # Apply the neural network
        z = z.permute(0, 3, 1, 2)  # [batch_size*num_tx, channels, height, width]
        for conv in self.hidden_conv:
            z = conv(z)
        z = self.output_conv(z)

        # Skip connection
        z = z + s.permute(0, 3, 1, 2)

        # Unflatten
        z = z.permute(0, 2, 3, 1)  # [batch_size*num_tx, height, width, channels]
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new  # Updated tensor state for each user


class CGNNIt(nn.Module):
    """
    Implements an iteration of the CGNN detector.
    Consists of two stages: State aggregation followed by state update.

    Parameters
    ----------
    d_s : int
        Size of the state vector.
    num_units_agg : list of int
        Number of units for the hidden dense layers of the aggregation network.
    num_units_state_update : list of int
        Number of units for the hidden separable convolutional layers of the state-update network.
    layer_type_dense: str
        Layer type of Dense layers. Dense is used for state aggregation.
    layer_type_conv: str
        Layer type of convolutional layers. CNNs are used for state updates.
    dtype: torch.dtype
        Dtype of the layer.

    Input
    -----
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        State vector.
    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], torch.Tensor
        Map showing the position of the nearest pilot for every user in time and frequency.
        This can be seen as a form of positional encoding.
    active_tx: [batch_size, num_tx], torch.Tensor
        Active user mask where each `0` indicates non-active users and `1` indicates an active user.

    Output
    ------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], torch.Tensor
        Updated channel state vector.
    """

    def __init__(
        self,
        d_s,
        num_units_agg,
        num_units_state_update,
        layer_type_dense="linear",
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


class AggregateUserStates(nn.Module):

    def __init__(self, d_s, num_units, layer_type="linear", dtype=torch.float32):
        super().__init__()

        if layer_type in ["linear", "dense"]:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(d_s, n, dtype=dtype) for n in num_units]
            )
            self.output_layer = nn.Linear(num_units[-1], d_s, dtype=dtype)
        elif layer_type == "conv1d":
            self.hidden_layers = nn.ModuleList(
                [nn.Conv1d(d_s, n, kernel_size=1, dtype=dtype) for n in num_units]
            )
            self.output_layer = nn.Conv1d(
                num_units[-1], d_s, kernel_size=1, dtype=dtype
            )
        else:
            raise NotImplementedError(
                f"Layer type '{layer_type}' is not supported. Use 'linear', 'dense', or 'conv1d'."
            )

        self.activation = nn.ReLU()
        self.layer_type = layer_type

    def forward(self, inputs):
        s, active_tx = inputs

        # Process s
        sp = s
        for layer in self.hidden_layers:
            if self.layer_type == "conv1d":
                sp = sp.transpose(1, 2)  # Adjust dimensions for Conv1d
            sp = self.activation(layer(sp))
            if self.layer_type == "conv1d":
                sp = sp.transpose(1, 2)  # Restore original dimensions

        sp = self.output_layer(sp)

        # Aggregate all states
        active_tx = expand_to_rank(active_tx, sp.dim(), axis=-1)
        sp = torch.mul(sp, active_tx)

        # Aggregate and remove self-state
        a = torch.sum(sp, dim=1, keepdim=True) - sp

        # Scale by number of active users
        p = torch.sum(active_tx, dim=1, keepdim=True) - 1.0
        p = torch.clamp(p, min=0.0)  # clip negative values to ignore non-active user
        p = torch.where(p == 0.0, torch.ones_like(p), 1.0 / p)

        # Scale states by number of aggregated users
        a = torch.mul(a, p)

        return a


class UpdateState(nn.Module):

    def __init__(self, d_s, num_units, layer_type="sepconv", dtype=torch.float32):
        super().__init__()

        if layer_type == "sepconv":
            conv_layer = lambda in_c, out_c: nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c),
                nn.Conv2d(in_c, out_c, kernel_size=1),
            )
        elif layer_type == "conv":
            conv_layer = lambda in_c, out_c: nn.Conv2d(
                in_c, out_c, kernel_size=3, padding=1
            )
        else:
            raise NotImplementedError(f"Unknown layer_type '{layer_type}' selected.")

        # Hidden blocks
        self.hidden_conv = nn.ModuleList()
        for n in num_units:
            self.hidden_conv.append(nn.Sequential(conv_layer(n, n), nn.ReLU()))

        # Output block
        self.output_conv = conv_layer(num_units[-1], d_s)
        # Convert all layers to specified dtype
        self = self.to(dtype)

    def forward(self, inputs):
        s, a, pe = inputs

        batch_size = s.shape[0]
        num_tx = s.shape[1]

        # Stack the inputs
        pe = pe.repeat(batch_size, 1, 1, 1, 1)
        pe = flatten_dims(pe, 2, 0)
        s = flatten_dims(s, 2, 0)
        a = flatten_dims(a, 2, 0)

        z = torch.cat([a, s, pe], dim=-1)

        # Apply the neural network
        z = z.permute(0, 3, 1, 2)  # [batch_size*num_tx, channels, height, width]
        for conv in self.hidden_conv:
            z = conv(z)
        z = self.output_conv(z)

        # Skip connection
        z = z + s.permute(0, 3, 1, 2)

        # Unflatten
        z = z.permute(0, 2, 3, 1)  # [batch_size*num_tx, height, width, channels]
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new


class ReadoutLLRs(nn.Module):
    """
    Network computing LLRs from the state vectors.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a linear layer without non-linearity and with
    `num_bits_per_symbol` units.

    Parameters
    ----------
    num_bits_per_symbol : int
        Number of bits per symbol.
    num_units : list of int
        Number of units for the hidden layers.
    layer_type : str, optional
        Defines which type of linear layers are used. Default is "linear".
    dtype : torch.dtype, optional
        Dtype of the layer. Default is torch.float32.

    Input
    -----
    s : torch.Tensor
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        Data state.

    Output
    ------
    torch.Tensor
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_bits_per_symbol]
        LLRs for each bit of each stream.
    """

    def __init__(
        self, num_bits_per_symbol, num_units, layer_type="linear", dtype=torch.float32
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        in_features = num_units[
            0
        ]  # Assume the input size is the first element of num_units

        for n in num_units[1:]:  # Start from the second element
            if layer_type == "linear":
                self.layers.append(nn.Linear(in_features, n, dtype=dtype))
            elif layer_type == "conv1d":
                self.layers.append(
                    nn.Conv1d(in_features, n, kernel_size=1, dtype=dtype)
                )
            elif layer_type == "conv2d":
                self.layers.append(
                    nn.Conv2d(in_features, n, kernel_size=1, dtype=dtype)
                )
            else:
                raise NotImplementedError(
                    f"Layer type '{layer_type}' is not supported. Use 'linear', 'conv1d', or 'conv2d'."
                )

            self.layers.append(nn.ReLU())
            in_features = n

        # Output layer
        if layer_type == "linear":
            self.output_layer = nn.Linear(
                num_units[-1], num_bits_per_symbol, dtype=dtype
            )
        elif layer_type == "conv1d":
            self.output_layer = nn.Conv1d(
                num_units[-1], num_bits_per_symbol, kernel_size=1, dtype=dtype
            )
        elif layer_type == "conv2d":
            self.output_layer = nn.Conv2d(
                num_units[-1], num_bits_per_symbol, kernel_size=1, dtype=dtype
            )

        self.layer_type = layer_type

    def forward(self, s):
        z = s
        if self.layer_type == "conv1d":
            z = z.transpose(1, 2)
        elif self.layer_type == "conv2d":
            z = z.permute(0, 4, 1, 2, 3)

        for layer in self.layers:
            z = layer(z)

        llr = self.output_layer(z)

        if self.layer_type == "conv1d":
            llr = llr.transpose(1, 2)
        elif self.layer_type == "conv2d":
            llr = llr.permute(0, 2, 3, 4, 1)

        return llr


class ReadoutChEst(nn.Module):
    """
    Network computing channel estimate.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a linear layer without non-linearity and with
    2*num_rx_ant units.

    Parameters
    ----------
    num_rx_ant : int
        Number of receive antennas.
    num_units : list of int
        Number of units for the hidden layers.
    layer_type : str, optional
        Defines which type of linear layers are used. Default is "linear".
    dtype : torch.dtype, optional
        Dtype of the layer. Default is torch.float32.

    Input
    -----
    s : torch.Tensor
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        State vector.

    Output
    ------
    torch.Tensor
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        Channel estimate for each stream.
    """

    def __init__(self, num_rx_ant, num_units, layer_type="linear", dtype=torch.float32):
        super().__init__()

        if layer_type in ["linear", "dense"]:
            self.layers = nn.ModuleList()
            for n in num_units:
                self.layers.append(nn.Linear(n, n, dtype=dtype))
                self.layers.append(nn.ReLU())
            self.output_layer = nn.Linear(num_units[-1], 2 * num_rx_ant, dtype=dtype)
        elif layer_type == "conv1d":
            self.layers = nn.ModuleList()
            for n in num_units:
                self.layers.append(nn.Conv1d(n, n, kernel_size=1, dtype=dtype))
                self.layers.append(nn.ReLU())
            self.output_layer = nn.Conv1d(
                num_units[-1], 2 * num_rx_ant, kernel_size=1, dtype=dtype
            )
        else:
            raise NotImplementedError(
                f"Layer type '{layer_type}' is not supported. Use 'linear', 'dense', or 'conv1d'."
            )

        self.layer_type = layer_type

    def forward(self, s):
        z = s
        if self.layer_type == "conv1d":
            z = z.transpose(1, 2)  # Adjust for Conv1d input shape

        for layer in self.layers:
            z = layer(z)

        z = self.output_layer(z)

        if self.layer_type == "conv1d":
            z = z.transpose(1, 2)  # Adjust back to original shape

        return z


class CGNN(nn.Module):
    """
    Implements the core neural receiver consisting of convolutional and graph layer components (CGNN).

    Parameters:
    -----------
    num_bits_per_symbol : list of ints
        Number of bits per resource element. Defined as list for mixed MCS schemes.
    num_rx_ant : int
        Number of receive antennas
    num_it : int
        Number of iterations.
    d_s : int
        Size of the state vector.
    num_units_init : list of int
        Number of hidden units for the init network.
    num_units_agg : list of list of ints
        Number of kernel for the hidden dense layers of the aggregation network per iteration.
    num_units_state : list of list of ints
        Number of hidden units for the state-update network per iteration.
    num_units_readout : list of int
        Number of hidden units for the read-out network.
    layer_type_dense: str
        Layer type of Dense layers.
    layer_type_conv: str
        Layer type of convolutional layers.
    layer_type_readout: str
        Layer type of Dense readout layers.
    training : boolean
        Set to `True` if instantiated for training. Set to `False` otherwise.
    apply_multiloss : boolean
        If True, apply loss at each iteration.
    var_mcs_masking : boolean
        If True, use variable MCS masking.
    dtype: torch.dtype
        Dtype of the layer.
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
    ):
        super().__init__()
        # Add this line
        self.iterations = nn.ModuleList(
            [
                CGNNIt(
                    d_s,
                    num_units_agg[i],
                    num_units_state[i],
                    layer_type_dense=layer_type_dense,
                    layer_type_conv=layer_type_conv,
                    dtype=dtype,
                )
                for i in range(num_it)
            ]
        )
        self.training = training
        self.apply_multiloss = apply_multiloss
        self.var_mcs_masking = var_mcs_masking
        self.num_it = num_it
        self.num_mcss_supported = len(num_bits_per_symbol)
        self.num_bits_per_symbol = num_bits_per_symbol

        # Initialization for the state
        if self.var_mcs_masking:
            self.s_init = StateInit(
                d_s, num_units_init, layer_type=layer_type_conv, dtype=dtype
            )
        else:
            self.s_init = nn.ModuleList(
                [
                    StateInit(
                        d_s, num_units_init, layer_type=layer_type_conv, dtype=dtype
                    )
                    for _ in num_bits_per_symbol
                ]
            )

        # Iterations blocks
        self.iterations = nn.ModuleList(
            [
                CGNNIt(
                    d_s,
                    num_units_agg[i],
                    num_units_state[i],
                    layer_type_dense=layer_type_dense,
                    layer_type_conv=layer_type_conv,
                    dtype=dtype,
                )
                for i in range(num_it)
            ]
        )

        # Readouts
        if self.var_mcs_masking:
            self.readout_llrs = ReadoutLLRs(
                max(num_bits_per_symbol),
                num_units_readout,
                layer_type=layer_type_readout,
                dtype=dtype,
            )
        else:
            self.readout_llrs = nn.ModuleList(
                [
                    ReadoutLLRs(
                        num_bits,
                        num_units_readout,
                        layer_type=layer_type_readout,
                        dtype=dtype,
                    )
                    for num_bits in num_bits_per_symbol
                ]
            )

        # self.readout_chest = ReadoutChEst(
        #     num_rx_ant, num_units_readout, layer_type=layer_type_readout, dtype=dtype
        # )
        self.readout_chest = ReadoutChEst(
            num_rx_ant, num_units_readout, layer_type="linear", dtype=dtype
        )

    def forward(self, inputs):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs
        # Ensure all inputs are PyTorch tensors with consistent shapes
        y = torch.as_tensor(y)
        pe = torch.as_tensor(pe)
        h_hat = torch.as_tensor(h_hat) if h_hat is not None else None
        active_tx = torch.as_tensor(active_tx)
        mcs_ue_mask = torch.as_tensor(mcs_ue_mask)
        # Print input shapes for debugging
        print(
            f"CGNN input shapes: y: {y.shape}, pe: {pe.shape}, h_hat: {h_hat.shape if h_hat is not None else 'None'}"
        )
        print(
            f"active_tx shape: {active_tx.shape}, mcs_ue_mask shape: {mcs_ue_mask.shape}"
        )

        # Normalization
        norm_scaling = torch.mean(torch.square(y), dim=(1, 2, 3), keepdim=True)
        norm_scaling = 1.0 / torch.sqrt(norm_scaling)
        y = y * norm_scaling
        norm_scaling = norm_scaling.unsqueeze(1)
        if h_hat is not None:
            h_hat = h_hat * norm_scaling

        # State initialization
        if self.var_mcs_masking:
            s = self.s_init((y, pe, h_hat))
        else:
            s = sum(
                self.s_init[idx]((y, pe, h_hat))
                * expand_to_rank(mcs_ue_mask[:, :, idx : idx + 1], 5, axis=-1)
                for idx in range(self.num_mcss_supported)
            )

        # Run receiver iterations
        llrs = []
        h_hats = []
        for i in range(self.num_it):
            # State update
            s = self.iterations[i]([s, pe, active_tx])

            # Read-outs
            if (self.training and self.apply_multiloss) or i == self.num_it - 1:
                llrs_ = []
                for idx in range(self.num_mcss_supported):
                    if self.var_mcs_masking:
                        llrs__ = self.readout_llrs(s)
                        llrs__ = llrs__[..., : self.num_bits_per_symbol[idx]]
                    else:
                        llrs__ = self.readout_llrs[idx](s)
                    llrs_.append(llrs__)
                llrs.append(llrs_)
                h_hats.append(self.readout_chest(s))

        return llrs, h_hats

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
            val <= len(self.iterations)
        ), "Invalid number of iterations"
        self._num_it = val


class CGNNOFDM(nn.Module):
    """
    Wrapper for the neural receiver (CGNN) layer that handles
    OFDM waveforms and the resourcegrid mapping/demapping.
    Layer also integrates loss function computation.

    Parameters:
    -----------
    sys_parameters : Parameters
        The system parameters.
    max_num_tx : int
        Maximum number of transmitters
    training : bool
        Set to `True` if instantiated for training. Set to `False` otherwise.
    num_it : int
        Number of iterations.
    d_s : int
        Size of the state vector.
    num_units_init : list of int
        Number of hidden units for the init network.
    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network.
    num_units_state : list of int
        Number of hidden units for the state-update network.
    num_units_readout : list of int
        Number of hidden units for the read-out network.
    layer_demappers : list of nn.Module
        List of layer demappers for each MCS.
    layer_type_dense: str
        Layer type of Dense layers.
    layer_type_conv: str
        Layer type of convolutional layers.
    layer_type_readout: str
        Layer type of Dense readout layers.
    dtype: torch.dtype
        DType of the NRX layers.
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
        layer_type_dense="linear",
        layer_type_conv="sepconv",
        layer_type_readout="linear",
        dtype=torch.float32,
    ):
        super().__init__()

        self.training = training
        self.max_num_tx = max_num_tx
        self.layer_demappers = layer_demappers
        self.sys_parameters = sys_parameters
        self.dtype = dtype
        self.num_mcss_supported = len(sys_parameters.mcs_index)
        self.rg = sys_parameters.transmitters[0]._resource_grid
        print("flag CGNNOFDM initialized")
        if self.sys_parameters.mask_pilots:
            print("Masking pilots for pilotless communications.")

        self.mcs_var_mcs_masking = False
        if hasattr(self.sys_parameters, "mcs_var_mcs_masking"):
            self.mcs_var_mcs_masking = self.sys_parameters.mcs_var_mcs_masking
            print("Var-MCS NRX with masking.")
        elif len(sys_parameters.mcs_index) > 1:
            print("Var-MCS NRX with MCS-specific IO layers.")

        num_bits_per_symbol = [
            sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol
            for mcs_list_idx in range(self.num_mcss_supported)
        ]

        num_rx_ant = sys_parameters.num_rx_antennas

        self.cgnn = CGNN(
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
            var_mcs_masking=self.mcs_var_mcs_masking,
            dtype=dtype,
        )

        self.rg_demapper = ResourceGridDemapper(self.rg, sys_parameters.sm)

        if training:
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.mse = nn.MSELoss(reduction="none")

        self.nearest_pilot_dist = self._compute_nearest_pilot_dist()

    def _compute_nearest_pilot_dist(self):
        # Compute and store the nearest pilot distance (positional encoding)
        # This method should be implemented to match the TensorFlow version's functionality
        # The result should be a tensor of shape [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
        pass

    def _compute_positional_encoding(self, num_tx, num_subcarriers, num_ofdm_symbols):
        pe = torch.zeros(num_tx, num_subcarriers, num_ofdm_symbols, 2)
        for tx in range(num_tx):
            for sc in range(num_subcarriers):
                for sym in range(num_ofdm_symbols):
                    pe[tx, sc, sym, 0] = sc / num_subcarriers
                    pe[tx, sc, sym, 1] = sym / num_ofdm_symbols
        return pe

    def forward(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None):
        print("flag CGNNOFDM forward")
        print(f"Input types: {[type(inp) for inp in inputs]}")
        print(
            f"Input shapes: {[inp.shape if hasattr(inp, 'shape') else 'scalar' for inp in inputs]}"
        )
        if self.training:
            print("flag 1")
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask = [
                ensure_torch_tensor(x) for x in inputs
            ]
        else:
            print("flag 2")
            y, h_hat_init, active_tx = [ensure_torch_tensor(x) for x in inputs]

        # Convert inputs to PyTorch tensors if they aren't already
        y = torch.as_tensor(y).to(self.dtype)
        h_hat_init = (
            torch.as_tensor(h_hat_init).to(self.dtype)
            if h_hat_init is not None
            else None
        )
        active_tx = torch.as_tensor(active_tx).to(self.dtype)
        print("flag 3")
        # Check if any of the tensors are scalars and handle accordingly
        if y.ndim == 0 or h_hat_init.ndim == 0 or active_tx.ndim == 0:
            raise ValueError(
                "One or more input tensors are scalars. Expected multi-dimensional tensors."
            )

        num_tx = active_tx.shape[1]
        num_subcarriers = y.shape[1]
        num_ofdm_symbols = y.shape[2]
        print("flag 3.1")

        if self.sys_parameters.mask_pilots:
            rg_type = self.rg.build_type_grid()
            rg_type = rg_type.unsqueeze(0).expand(y.shape)
            y = torch.where(rg_type == 1, torch.tensor(0.0, dtype=y.dtype), y)

        print("flag 3.2")
        y = y[:, 0]
        y = y.permute(0, 3, 2, 1)

        # Check if y is complex
        if y.is_complex():
            y = torch.cat([y.real, y.imag], dim=-1)
        else:
            # If y is already real-valued, reshape it to match the expected shape
            y = y.reshape(*y.shape[:-1], -1, 2)
            y = y.reshape(*y.shape[:-2], -1)

        # Compute positional encoding
        print("flag 3.3")
        pe = self._compute_positional_encoding(
            num_tx, num_subcarriers, num_ofdm_symbols
        )
        pe = pe.to(self.dtype).to(y.device)
        # Ensure mcs_ue_mask is properly initialized
        print("flag 3.4")
        if mcs_ue_mask_eval is None:
            mcs_ue_mask = torch.nn.functional.one_hot(
                torch.tensor(mcs_arr_eval[0]), num_classes=self.num_mcss_supported
            ).to(y.device)
        else:
            mcs_ue_mask = ensure_torch_tensor(mcs_ue_mask_eval).to(y.device)
        mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)

        print("flag 3.7")
        llrs_, h_hats_ = self.cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask])

        print("flag 3.71")
        indices = mcs_arr_eval
        llrs = []
        h_hats = []
        print("flag 3.8")
        for llrs_, h_hat_ in zip(llrs_, h_hats_):
            h_hat_ = h_hat_.float()
            _llrs_ = []
            print("flag 3.9")
            for idx in indices:
                llrs_[idx] = llrs_[idx].float()
                llrs_[idx] = llrs_[idx].permute(0, 1, 3, 2, 4).unsqueeze(1)
                llrs_[idx] = self.rg_demapper(llrs_[idx])
                llrs_[idx] = llrs_[idx][:, :num_tx]
                llrs_[idx] = torch.flatten(llrs_[idx], start_dim=-2)
                print("flag 3.10")
                if self.layer_demappers is None:
                    print("flag 3.11")
                    llrs_[idx] = llrs_[idx].squeeze(-2)
                else:
                    print("flag 3.12")
                    llrs_[idx] = self.layer_demappers[idx](llrs_[idx])

                _llrs_.append(llrs_[idx])
            print("flag 3.13")
            llrs.append(_llrs_)
            h_hats.append(h_hat_)

        if self.training:
            print("flag 3.14")
            loss_data = torch.tensor(0.0, dtype=torch.float32)
            print("flag 3.15")
            for llrs_ in llrs:
                for idx, llr in enumerate(llrs_):
                    print("flag 3.16")
                    loss_data_ = self.bce(llr, bits[idx])
                    mcs_ue_mask_ = expand_to_rank(
                        mcs_ue_mask[:, :, indices[idx] : indices[idx] + 1],
                        loss_data_.dim(),
                        axis=-1,
                    )
                    loss_data_ = loss_data_ * mcs_ue_mask_
                    active_tx_data = expand_to_rank(
                        active_tx, loss_data_.dim(), axis=-1
                    )
                    loss_data_ = loss_data_ * active_tx_data
                    loss_data += loss_data_.mean()

            loss_chest = torch.tensor(0.0, dtype=torch.float32)
            if h_hats is not None and h is not None:
                for h_hat_ in h_hats:
                    loss_chest += self.mse(h, h_hat_)
                active_tx_chest = expand_to_rank(active_tx, loss_chest.dim(), axis=-1)
                loss_chest = loss_chest * active_tx_chest
                loss_chest = loss_chest.mean()

            return loss_data, loss_chest
        else:
            return llrs[-1][0], h_hats[-1]


class TBEncoderWrapper(nn.Module):
    def __init__(self, tb_encoder):
        super().__init__()
        self.tb_encoder = tb_encoder

    def forward(self, x):
        # Convert PyTorch tensor to NumPy array
        x_np = x.detach().cpu().numpy()

        # Use Sionna's TBEncoder
        y_tf = self.tb_encoder(x_np)

        # Convert TensorFlow tensor back to PyTorch tensor
        y = torch.from_numpy(y_tf.numpy()).to(x.device)

        return y


class TBDecoderWrapper(nn.Module):
    def __init__(self, tb_decoder):
        super().__init__()
        self.tb_decoder = tb_decoder

    def forward(self, x):
        # Convert PyTorch tensor to NumPy array
        x_np = x.detach().cpu().numpy()

        # Use Sionna's TBDecoder
        y_tf = self.tb_decoder(x_np)

        # Convert TensorFlow tensor back to PyTorch tensor
        y = torch.from_numpy(y_tf.numpy()).to(x.device)

        return y


class LayerDemapperWrapper(nn.Module):
    def __init__(self, layer_mapper, num_bits_per_symbol):
        super().__init__()
        self.layer_demapper = SionnaLayerDemapper(layer_mapper, num_bits_per_symbol)

    def forward(self, x):
        # Convert PyTorch tensor to NumPy array
        x_np = x.detach().cpu().numpy()

        # Use Sionna's LayerDemapper
        y_tf = self.layer_demapper(x_np)

        # Convert TensorFlow tensor back to PyTorch tensor
        y = torch.from_numpy(y_tf.numpy()).to(x.device)

        return y


class NeuralPUSCHReceiver(nn.Module):
    def __init__(self, sys_parameters, training=False):
        print("Flag: Neural receiver -> __init__")
        super().__init__()
        self._sys_parameters = sys_parameters
        self._training = training

        # Initialize transport block encoders and decoders
        self._tb_encoders = nn.ModuleList()
        self._tb_decoders = nn.ModuleList()
        print("Flag: 0")
        self._num_mcss_supported = (
            sys_parameters.mcs_index.shape[0]
            if isinstance(sys_parameters.mcs_index, tf.Tensor)
            else len(sys_parameters.mcs_index)
        )
        print("Flag: 0.1")
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
        print("Flag: 0.2")
        # Precoding matrix
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            print("Flag: 0.31")
            precoder_w = sys_parameters.transmitters[0]._precoder._w
            if isinstance(precoder_w, (tf.Tensor, np.ndarray)):
                # If it's already a tensor or numpy array, convert to PyTorch tensor
                self._precoding_mat = torch.tensor(
                    precoder_w.numpy()
                    if isinstance(precoder_w, tf.Tensor)
                    else precoder_w
                )
            elif np.isscalar(precoder_w):
                # If it's a scalar, create a 1x1 tensor
                self._precoding_mat = torch.tensor([[precoder_w]])
            else:
                # If it's something else (like a list), convert to tensor normally
                self._precoding_mat = torch.tensor(precoder_w)
            print("Flag: 0.3")
        else:
            print("Flag: 0.32")
            self._precoding_mat = torch.ones(
                sys_parameters.max_num_tx,
                sys_parameters.num_antenna_ports,
                1,
                dtype=torch.complex64,
            )
            print("Flag: 0.4")

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
        print("Flag: 0.5")
        rg_type = rg.build_type_grid()[:, 0]
        # Convert TensorFlow tensor to NumPy array, then to PyTorch tensor
        rg_type_torch = torch.from_numpy(rg_type.numpy())
        pilot_ind = torch.where(rg_type_torch == 1)
        self._pilot_ind = pilot_ind[0]

        # Layer demappers
        self._layer_demappers = nn.ModuleList()
        for mcs_list_idx in range(self._num_mcss_supported):
            self._layer_demappers.append(
                LayerDemapperWrapper(
                    sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                    sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol,
                )
            )
        print("Flag: 0.6")
        # Neural receiver
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
            dtype=getattr(torch, sys_parameters.nrx_dtype.name),
        )

    def estimate_channel(self, y, num_tx):
        if isinstance(y, torch.Tensor):
            y_tf = tf.convert_to_tensor(y.detach().cpu().numpy())
        else:
            y_tf = y

        if self._sys_parameters.initial_chest == "ls":
            if self._sys_parameters.mask_pilots:
                raise ValueError(
                    "Cannot use initial channel estimator if pilots are masked."
                )
            h_hat, _ = self._ls_est([y_tf, tf.constant(1e-1)])
            h_hat = h_hat[:, 0, :, :num_tx, 0]
            h_hat = tf.transpose(h_hat, perm=[0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)

            # Convert TensorFlow tensor back to PyTorch tensor
            h_hat = torch.from_numpy(h_hat.numpy())
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
        print("Flag: Neural receiver -> Forward")
        if self._training:
            y, active_tx, b, h, mcs_ue_mask = inputs
            if isinstance(mcs_arr_eval, tf.Tensor):
                mcs_arr_eval = mcs_arr_eval.numpy().tolist()
            if len(mcs_arr_eval) == 1 and not isinstance(b, list):
                b = [b]
            bits = []
            for idx, mcs in enumerate(mcs_arr_eval):
                bits.append(self._sys_parameters.transmitters[mcs]._tb_encoder(b[idx]))

            num_tx = active_tx.shape[1]
            h_hat = self.estimate_channel(y, num_tx)
            print("Flag: 1")
            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            losses = self._neural_rx(
                (y, h_hat, active_tx, bits, h, mcs_ue_mask), mcs_arr_eval
            )
            print("Flag: 2")
            return losses
        else:
            print("Flag: 3")
            y, active_tx = inputs
            num_tx = active_tx.shape[1]

            # Ensure inputs are PyTorch tensors
            y = torch.as_tensor(y)
            active_tx = torch.as_tensor(active_tx)

            print("Flag: 3.1")
            # Estimate channel using PyTorch tensors
            h_hat = self.estimate_channel(y, num_tx)

            print("Flag: 3.3")
            # Call _neural_rx with PyTorch tensors
            llr, h_hat_refined = self._neural_rx(
                (y, h_hat, active_tx),
                [mcs_arr_eval[0]],
                mcs_ue_mask_eval=mcs_ue_mask_eval,
            )

            print("Flag: 4")
            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)
            return b_hat, h_hat_refined, h_hat, tb_crc_status


################################
## ONNX Layers / Wrapper
################################
# The following layers provide an adapter to the Aerial PUSCH pipeline
# the code is only relevant for for ONNX/TensorRT exports but can be ignored
# for Sionna-based simulations.


class NRPreprocessing(nn.Module):
    """
    Pre-preprocessing layer for the neural receiver applying initial channel estimation.

    This layer takes the channel estimates at pilot positions and performs nearest neighbor
    interpolation on a "per PRB" basis. It scales to arbitrary resource grid sizes.

    The input/output shapes are Aerial compatible and not directly compatible with Sionna.
    The returned "resourcegrid of LLRs" can be further processed using PyAerial.

    Note that all operations are real-valued.

    Parameters:
    -----------
    num_tx : int
        Number of transmitters (i.e., independent streams)

    Input:
    ------
    y : torch.Tensor
        [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], float32
        The received OFDM resource grid after cyclic prefix removal and FFT.
        Real and imaginary parts are stacked in the rx_antenna direction.
    h_hat : torch.Tensor
        [bs, num_pilots, num_streams, 2*num_rx_ant], float32
        Channel estimates at pilot positions. Real and imaginary parts are stacked in the rx_antenna direction.
    dmrs_ofdm_pos : torch.Tensor
        [num_tx, num_dmrs_symbols], int
        DMRS symbol positions within slot.
    dmrs_subcarrier_pos : torch.Tensor
        [num_tx, num_pilots_per_PRB], int
        Pilot position per PRB.

    Output:
    -------
    h_hat : torch.Tensor
        [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], float
        Channel estimate after nearest neighbor interpolation.
    pe : torch.Tensor
        [num_tx, num_subcarriers, num_ofdm_symbols, 2], float
        Map showing the position of the nearest pilot for every user in time and frequency.
        This can be seen as a form of positional encoding.
    """

    def __init__(self, num_tx):
        super().__init__()
        self._num_tx = num_tx
        self._num_res_per_prb = 12  # fixed in 5G

    def _focc_removal(self, h_hat):
        """Apply FOCC removal to h_hat."""
        shape = [-1, 2]
        s = h_hat.shape
        new_shape = s[:3] + shape
        h_hat = h_hat.reshape(new_shape)
        h_hat = torch.sum(h_hat, dim=-1, keepdim=True) / 2.0
        h_hat = h_hat.repeat(1, 1, 1, 1, 2)
        h_ls = h_hat.reshape(s[:3] + (-1,))
        return h_ls

    def _calculate_nn_indices(
        self, dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
    ):
        """Calculate nearest neighbor interpolation indices for a single PRB."""
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
            pilot_pos = torch.stack(
                torch.meshgrid(dmrs_subcarrier_pos[tx_idx], dmrs_ofdm_pos[tx_idx]),
                dim=-1,
            )
            pilot_pos = pilot_pos.reshape(1, -1, 2)

            diff = torch.abs(re_pos - pilot_pos)
            dist = torch.sum(diff, dim=-1)
            nn_idx = torch.argmin(dist, dim=1)
            nn_idx = nn_idx.reshape(1, 1, num_ofdm_symbols, self._num_res_per_prb)

            pe = torch.min(diff, dim=1)[0]
            pe = pe.reshape(1, num_ofdm_symbols, self._num_res_per_prb, 2)
            pe = pe.permute(0, 2, 1, 3)

            pe = pe.float()
            p = []
            for i in range(2):
                pe_ = pe[..., i : i + 1]
                pe_ -= pe_.mean()
                std_ = pe_.std()
                pe_ = torch.where(std_ > 0, pe_ / std_, pe_)
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
        """Apply nearest neighbor interpolation of pilots to all data symbols in the resource grid."""
        num_pilots_per_dmrs = dmrs_subcarrier_pos.shape[1]
        num_prbs = h_hat.shape[-1] // (num_pilots_per_dmrs * dmrs_ofdm_pos.shape[-1])

        s = h_hat.shape
        h_hat = h_hat.reshape(s[:3] + (-1, num_pilots_per_dmrs))
        h_hat = h_hat.permute(0, 1, 2, 4, 3)
        h_hat = h_hat.reshape(s[:3] + (-1,))

        nn_idx, pe = self._calculate_nn_indices(
            dmrs_ofdm_pos, dmrs_subcarrier_pos, num_ofdm_symbols, num_prbs
        )

        h_hat = (
            h_hat.unsqueeze(3)
            .unsqueeze(4)
            .repeat(1, 1, 1, num_ofdm_symbols, self._num_res_per_prb * num_prbs, 1)
        )
        nn_idx = nn_idx.unsqueeze(0).repeat(s[0], 1, 1, 1, num_prbs)
        h_hat = torch.gather(
            h_hat, dim=-1, index=nn_idx.unsqueeze(-1).repeat(1, 1, 1, 1, 1, s[1])
        )

        h_hat = h_hat.permute(0, 2, 3, 4, 1, 5)
        h_hat = h_hat.reshape(
            s[0],
            self._num_tx,
            1,
            num_ofdm_symbols,
            self._num_res_per_prb * num_prbs,
            s[1],
        )

        return h_hat, pe

    def forward(self, y, h_hat, dmrs_ofdm_pos, dmrs_subcarrier_pos):
        """Forward pass of the NRPreprocessing layer."""
        num_ofdm_symbols = y.shape[2]
        h_hat = self._focc_removal(h_hat)
        h_hat, pe = self._nn_interpolation(
            h_hat, num_ofdm_symbols, dmrs_ofdm_pos, dmrs_subcarrier_pos
        )
        h_hat = h_hat.squeeze(2)
        h_hat = h_hat.permute(0, 1, 3, 2, 4)
        return h_hat, pe


class NeuralReceiverONNX(Model):
    # pylint: disable=line-too-long
    r"""
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

    Input
    ------
    (rx_slot, h_hat, dmrs_port_mask)
    Tuple:

    rx_slot_real : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Real part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    rx_slot_imag : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Imaginary part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    h_hat_real : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Real part of the LS channel estimates at pilot positions.

    h_hat_imag : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Imaginary part of the LS channel estimates at pilot positions.

    dmrs_port_mask: [bs, num_tx], tf.float32
        Mask of 0s and 1s to indicate that DMRS ports are active or not.

    dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
        DMRS symbol positions within slot.

    dmrs_subcarrier_pos: [num_tx, num_pilots_per_prb], tf.int
        Pilot positions per PRB.

    Output
    -------
    (llr, h_hat)
    Tuple:

    llr : [batch_size, num_bits_per_symbol, num_tx, num_effective_subcarriers,
          num_ofdm_symbols], tf.float
        LLRs on bits.

    h_hat : [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], tf.float
        Refined channel estimates.
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

        super().__init__(**kwargs)
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

    def call(self, inputs):

        (
            y_real,
            y_imag,
            h_hat_real,
            h_hat_imag,
            dmrs_port_mask,
            dmrs_ofdm_pos,
            dmrs_subcarrier_pos,
        ) = inputs

        y = tf.concat((y_real, y_imag), axis=-1)
        h_hat_p = tf.concat((h_hat_real, h_hat_imag), axis=-1)

        # nearest neighbor interpolation of channel estimates
        h_hat, pe = self._preprocessing(
            (y, h_hat_p, dmrs_ofdm_pos, dmrs_subcarrier_pos)
        )

        # dummy MCS mask (no support for mixed MCS)
        mcs_ue_mask = tf.ones((1, 1, 1), tf.float32)

        # and run NRX
        llr, h_hat = self._cgnn([y, pe, h_hat, dmrs_port_mask, mcs_ue_mask])

        # cgnn returns list of results for each iteration
        # (not needed for inference)
        llr = llr[-1][0]  # take LLRs of first MCS (no support for mixed MCS)
        h_hat = h_hat[-1]

        # cast back to tf.float32 (if NRX uses quantization)
        llr = tf.cast(llr, tf.float32)
        h_hat = tf.cast(h_hat, tf.float32)

        # reshape llrs in Aerial format
        llr = tf.transpose(llr, (0, 4, 1, 2, 3))
        # Sionna defines LLRs with different sign
        llr = -1.0 * llr

        return llr, h_hat
