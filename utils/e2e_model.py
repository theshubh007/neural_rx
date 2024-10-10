# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

##### E2E model for system evaluations #####

import torch
import torch.nn as nn
import numpy as np
from sionna.utils import BinarySource, expand_to_rank
from .baseline_rx import BaselineReceiver
from .neural_rx import NeuralPUSCHReceiver
from sionna.channel import AWGN


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


class E2E_Model(nn.Module):
    r"""E2E_Model(sys_parameters, training=False, return_tb_status=False, **kwargs)
    End-to-end model for system evaluation.

    The model can be used with the neural receiver as well as with the baseline
    receivers.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training : bool
        If True, the model returns the loss instead of b and b_hat.

    return_tb_status : bool
        If True, the model returns if the transport block CRC holds.

    mcs_arr_eval_idx : int
        Selects the element (index) of the mcs_index array to evaluate. Can be
        overwritten as call argument.
        Defaults to 0, which selects the very first element specified in the
        mcs_index array.

    Input
    -----
    batch_size : int
        Batch size of random transmit signals to be generated

    ebno_db: float
        SNR in dB. Interpreted as rate-adjusted SNR if ``sys_parameters.ebno``
        is `True`.

    num_tx : int | None
        Number of active DMRS ports. Remark: the model always simulates all
        max_num_tx UEs. However, they are masked at the tx output, i.e., no
        energy is transmitted. This allows training with varying number of
        users without re-building the TensorFlow graph.

    output_nrx_h_hat : bool
        If True, the NRX internal channel estimations are returned.
        This is required for training with double read-outs.

    mcs_arr_eval_idx : int
        Selects the element (index) of the mcs_index array to evaluate.
        If not specified (defaults to None), the index set by the constructor
        is used. Has no effect if an mcs_ue_mask is specified.

    mcs_ue_mask : None, or [batch_size, max_num_tx, len(mcs_index)], tf.int32
        One-hot mask that specifies the MCS index of each UE for each batch
        sample. If not specified (defaults to None), the mcs_ue_mask will be
        inferred from msc_arr_eval_index.

    active_dmrs : None, or [batch_size, max_num_tx], tf.int32
        Optional one-hot mask that specifies the active DMRS for each batch
        sample. If active_dmrs is None (default), the E2E model generates the
        active_dmrs mask by itself.

    Output
    ------
    In inference mode (training=False)
        Depending on if return_tb_status and output_nrx_h_hat is True

        Tuple : b, b_hat, [tb_crc_status], [h, h_hat_refined, h_hat]

            b : [batch_size, num_tx, tb_size], tf.float
                    Transmitted information bits for selected MCS.

            b_hat : [batch_size, num_tx, tb_size], tf.float
                Decoded information bits for selected MCS.

            tb_crc_status: [batch_size, num_tx], tf.float
                Status of the TB CRC for each decoded TB.

            h: [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Ground truth channel CFRs

            h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
                Refined channel estimate from the NRX.

            h_hat, [batch_size, num_tx, num_effective_subcarriers,
                   num_ofdm_symbols, 2*num_rx_ant]
                Initial channel estimate used for the NRX.

    In training mode:
        losses: tuple with (loss_data, loss_chest) from NeuralPUSCHReceiver
            Only if ``training`` is `True`.

            loss_data: tf.float
                Binary cross-entropy loss on LLRs. Computed from active UEs and
                their selected MCSs.

            loss_chest: tf.float
                Mean-squared-error (MSE) loss between channel estimates and
                ground truth channel CFRs. Only relevant if double-readout is
                used.

    """

    def __init__(
        self, sys_parameters, training=False, return_tb_status=False, mcs_arr_eval_idx=0
    ):

        super().__init__()

        assert isinstance(
            mcs_arr_eval_idx, int
        ), "E2E Model can only evaluate one MCS at a time. For mixed MCS evaluation, use the E2E_Model_Mixed_MCS class."

        self._sys_parameters = sys_parameters
        self._training = training
        self._return_tb_status = return_tb_status
        self._mcs_arr_eval_idx = mcs_arr_eval_idx
        self.num_rx_antennas = sys_parameters.num_rx_antennas  # Add this line

        ###################################
        # Transmitter
        ###################################

        self._source = BinarySource()
        self._transmitters = sys_parameters.transmitters

        ###################################
        # Channel
        ###################################
        self._channel = sys_parameters.channel

        ###################################
        # Receiver
        ###################################

        if self._sys_parameters.system == "baseline_perf_csi_kbest":
            self._sys_name = "Baseline - Perf. CSI & K-Best"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "baseline_perf_csi_lmmse":
            self._sys_name = "Baseline - Perf. CSI & LMMSE"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "baseline_lmmse_kbest":
            self._sys_name = "Baseline - LMMSE+K-Best"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "baseline_lmmse_lmmse":
            self._sys_name = "Baseline - LMMSE+LMMSE"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "baseline_lsnn_lmmse":
            self._sys_name = "Baseline - LS/nn+LMMSE"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "baseline_lslin_lmmse":
            self._sys_name = "Baseline - LS/lin+LMMSE"
            self._receiver = BaselineReceiver(
                self._sys_parameters,
                return_tb_status=return_tb_status,
                mcs_arr_eval_idx=mcs_arr_eval_idx,
            )

        elif self._sys_parameters.system == "nrx":
            self._sys_name = "Neural Receiver"
            self._receiver = NeuralPUSCHReceiver(self._sys_parameters, training)
        else:
            raise NotImplementedError("Unknown system selected!")

    def _active_dmrs_mask(self, batch_size, num_tx, max_num_tx):
        """Sample mask of num_tx active DMRS ports (=users/streams).
        Draws different realization per batch sample.

        Input
        -----
        batch_size: int
            Batchsize of mask

        num_tx: int
            Number of active DMRS ports.

        max_num_tx: int
            Total number of DMRS ports.

        Output
        ------
        dmrs_mask: [batch_size, max_num_tx], torch.float32
            Mask of randomly activated DMRS ports.
        """

        max_num_tx = int(max_num_tx)
        num_tx = int(num_tx)

        # Create a range tensor and expand it to batch size
        r = torch.arange(max_num_tx, dtype=torch.int32)
        r = r.unsqueeze(0).expand(batch_size, -1)

        # Create initial mask
        x = torch.where(r < num_tx, torch.ones_like(r), torch.zeros_like(r))

        # Shuffle each row independently
        x_p = torch.stack([row[torch.randperm(max_num_tx)] for row in x])

        return x_p.float()

    def _mask_active_dmrs(
        self, b, b_hat, num_tx, active_dmrs, mcs_arr_eval_idx, tb_crc_status=None
    ):
        """Remove inactive users/layers from b and b_hat"""
        batch_size = b.shape[0]

        # only focus on active users
        a_mask = expand_to_rank(active_dmrs, b_hat.dim(), dim=-1)
        a_mask = a_mask.expand_as(b_hat)

        b_hat = b_hat[a_mask]
        b_hat = b_hat.reshape(
            batch_size, num_tx, self._transmitters[mcs_arr_eval_idx]._tb_size
        )

        b = b[a_mask]
        b = b.reshape(batch_size, num_tx, self._transmitters[mcs_arr_eval_idx]._tb_size)

        if tb_crc_status is not None:
            a_mask = expand_to_rank(active_dmrs, tb_crc_status.dim(), dim=-1)
            a_mask = a_mask.expand_as(tb_crc_status)
            tb_crc_status = tb_crc_status[a_mask]
            tb_crc_status = tb_crc_status.reshape(batch_size, num_tx)
            return b, b_hat, tb_crc_status

        return b, b_hat

    def _set_transmitter_random_pilots(self):
        """
        Sample a random slot number and assign its pilots to the transmitter
        """
        pilot_set = self._sys_parameters.pilots
        num_pilots = pilot_set.size(0)
        random_pilot_ind = torch.randint(0, num_pilots, (1,))
        pilots = pilot_set[random_pilot_ind]
        for mcs_list_idx in range(len(self._sys_parameters.mcs_index)):
            self._transmitters[mcs_list_idx].pilot_pattern.pilots = pilots

    def forward(
        self,
        batch_size,
        ebno_db,
        num_tx=None,
        output_nrx_h_hat=False,
        mcs_arr_eval_idx=None,
        mcs_ue_mask=None,
        active_dmrs=None,
    ):
        """Defines end-to-end system model."""
        print("Forward: e2e_model")

        # Randomly sample num_tx active DMRS ports
        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx

        # If nothing is specified, select one pre-specified MCS
        if mcs_arr_eval_idx is None:
            mcs_arr_eval_idx = self._mcs_arr_eval_idx

        # Generate active DMRS mask (if not specified)
        if active_dmrs is None:
            active_dmrs = self._active_dmrs_mask(
                batch_size, num_tx, self._sys_parameters.max_num_tx
            )

        if mcs_ue_mask is None:
            # No MCS-to-UE-mask specified --> evaluate pre-specified MCS only
            assert isinstance(
                mcs_arr_eval_idx, int
            ), "Pre-defined MCS UE mask only works if mcs_arr_eval_idx is an integer"
            mcs_ue_mask = (
                torch.nn.functional.one_hot(
                    torch.tensor(mcs_arr_eval_idx),
                    num_classes=len(self._sys_parameters.mcs_index),
                )
                .unsqueeze(0)
                .repeat(batch_size, self._sys_parameters.max_num_tx, 1)
            )
            mcs_arr_eval = [mcs_arr_eval_idx]
        else:
            # MCS_UE_mask is not none --> we now need to process all MCSs
            if isinstance(mcs_arr_eval_idx, (list, tuple)):
                # Some different order specified. Useful to evaluate mixed MCS scenarios.
                assert len(mcs_arr_eval_idx) == len(
                    self._sys_parameters.mcs_index
                ), "MCS array size mismatch."
                mcs_arr_eval = mcs_arr_eval_idx
            else:
                # Process in order of MCS index array
                mcs_arr_eval = list(range(len(self._sys_parameters.mcs_index)))

        ###################################
        # Transmitters
        # One transmitter for each supported MCS
        ###################################
        print("flag1")

        b = []
        for idx in range(len(mcs_arr_eval)):
            tb_size = self._transmitters[mcs_arr_eval[idx]]._tb_size
            b.append(
                torch.zeros([batch_size, self._sys_parameters.max_num_tx, tb_size])
            )

        print("flag1.1")

        # Sample a random slot number and assign its pilots to the transmitter
        if self._training:
            self._set_transmitter_random_pilots()

        print("flag1.2")
        import tensorflow as tf

        # Combine transmit signals from all MCSs
        x = torch.zeros_like(b[0], dtype=torch.complex64)
        for idx in range(len(mcs_arr_eval)):
            _mcs_ue_mask = (
                mcs_ue_mask[:, :, mcs_arr_eval[idx]].unsqueeze(-1).expand_as(x)
            )

            # Debugging: Check if the transmitter is callable and print its type
            print(
                f"Type of transmitter {idx}: {type(self._transmitters[mcs_arr_eval[idx]])}"
            )

            # Debugging: Check shape of input
            print(f"Shape of b[idx]: {b[idx].shape}")

            try:

                # Convert PyTorch tensor to NumPy, then to TensorFlow tensor
                input_np = b[idx].cpu().numpy()
                input_tf = tf.convert_to_tensor(input_np)

                # Call the TensorFlow transmitter
                output_tf = self._transmitters[mcs_arr_eval[idx]](input_tf)

                # Convert TensorFlow tensor to NumPy and then to PyTorch tensor
                output_np = output_tf.numpy()
                output_torch = torch.from_numpy(output_np).to(b[idx].device)

                print(
                    f"Shape of transmitter output for MCS {mcs_arr_eval[idx]}: {output_torch.shape}"
                )

                # Proceed if everything works
                x += _mcs_ue_mask * output_torch

            except Exception as e:
                print(f"Error calling transmitter: {e}")
            print("Flag1.2.3")

        # Mask non-active DMRS ports by multiplying with 0
        print("flag1.3")
        active_tx_mask = active_dmrs.unsqueeze(-1).expand_as(x)
        x = x * active_tx_mask

        ###################################
        # Channel
        ###################################
        print("flag2")

        if self._sys_parameters.ebno:
            print("flag2.1")
            tx = self._sys_parameters.transmitters[0]

            # Convert num_pilots and num_res to NumPy arrays if needed
            num_pilots = tx._resource_grid.num_pilot_symbols
            num_pilots = np.array(num_pilots, dtype=np.float32)
            num_res = tx._resource_grid.num_resource_elements
            num_res = np.array(num_res, dtype=np.float32)

            print(f"Number of pilots: {num_pilots}{type(num_pilots)}")
            print(f"Number of resource elements: {num_res}{type(num_res)}")

            # Calculate ebno using NumPy
            ebno_db = ebno_db - 10.0 * np.log10(1.0 - num_pilots / num_res)

            # Ensure mcs_arr_eval_idx is treated correctly
            if isinstance(mcs_arr_eval_idx, int):  # If it's an integer
                mcs_idx = mcs_arr_eval_idx
            elif isinstance(mcs_arr_eval_idx, (list, tuple)):  # If it's a list or tuple
                mcs_idx = mcs_arr_eval_idx[0]
            else:
                raise TypeError(
                    "mcs_arr_eval_idx should be either an int or a list/tuple"
                )

            # Manually perform ebnodb2no logic using NumPy
            ebno = np.power(10.0, ebno_db / 10.0)
            num_bits_per_symbol = self._transmitters[mcs_idx]._num_bits_per_symbol
            coderate = self._transmitters[mcs_idx]._target_coderate
            energy_per_symbol = 1
            print(f"Energy per symbol: {energy_per_symbol}")
            no = 1 / (ebno * coderate * num_bits_per_symbol / energy_per_symbol)

            # Convert the result `no` back to a PyTorch tensor
            no = torch.tensor(no, dtype=torch.float32)
        else:
            no = torch.pow(10.0, -ebno_db / 10.0)

        # Check the result of noise calculation
        print(f"Noise Power Density (no): {no}")
        print("flag2.2")
        print(type(no))
        print(type(x))
        x = to_numpy(x)
        tf_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.complex(tf_tensor, tf.zeros_like(tf_tensor))
        print(type(x))
        no = to_numpy(no)
        tf_tensor = tf.convert_to_tensor(no, dtype=tf.float32)
        no = tf_tensor

        if self._sys_parameters.channel_type == "AWGN":
            # AWGN channel: pass both x and no
            print("flag2.33")

            y = self._channel([x, no])
            # Convert TensorFlow tensor y back to NumPy and then to PyTorch
            y_numpy = to_numpy(y)
            y_torch = torch.from_numpy(y_numpy)  # Convert to PyTorch tensor
            h = torch.ones_like(y_torch)  # Simple AWGN channel with no channel response
        else:

            print("flag2.5")
            self._channel = AWGN()
            y = self._channel([x, no])
            # Convert TensorFlow tensor y back to PyTorch
            y_numpy = to_numpy(y)
            y_torch = torch.from_numpy(y_numpy)
            h = torch.ones_like(y_torch)  # Simple AWGN channel with no channel

        ###################################
        # Receiver
        ###################################
        print("flag3")
        y = to_numpy(y)
        y = torch.from_numpy(y)
        h = to_numpy(h)
        h = torch.from_numpy(h)
        print(self._sys_parameters.system)

        if self._sys_parameters.system == "nrx":
            print("flag3.1")
            if self._training:
                print("flag3.1.1")
                # Return losses during training phase
                losses = self._receiver(y, active_dmrs, b, h, mcs_ue_mask, mcs_arr_eval)
                return losses
            else:
                # Return the results during evaluation phase
                print("flag3.1.2")
                print(
                    type(y),
                    type(active_dmrs),
                    type(b),
                    type(h),
                    type(mcs_ue_mask),
                    type(mcs_arr_eval),
                )
                b_hat, h_hat_refined, h_hat, tb_crc_status = self._receiver(
                    (y, active_dmrs), mcs_arr_eval, mcs_ue_mask_eval=mcs_ue_mask
                )
                return b, b_hat, h_hat_refined, h_hat

    # def forward(
    #     self,
    #     batch_size,
    #     ebno_db,
    #     num_tx=None,
    #     output_nrx_h_hat=False,
    #     mcs_arr_eval_idx=None,
    #     mcs_ue_mask=None,
    #     active_dmrs=None,
    # ):
    #     """defines end-to-end system model."""

    #     # randomly sample num_tx active dmrs ports
    #     # Convert inputs to PyTorch tensors if they're not already
    #     ebno_db = torch.tensor(ebno_db, dtype=torch.float32)

    #     if num_tx is None:
    #         num_tx = self._sys_parameters.max_num_tx

    #     # if nothing is specified, select one pre-specified MCS
    #     if mcs_arr_eval_idx is None:
    #         mcs_arr_eval_idx = self._mcs_arr_eval_idx

    #     # generate active dmrs mask (if not specified)
    #     if active_dmrs is None:
    #         active_dmrs = self._active_dmrs_mask(
    #             batch_size, num_tx, self._sys_parameters.max_num_tx
    #         )

    #     if mcs_ue_mask is None:
    #         # No mcs-to-ue-mask specified --> evaluate pre-specified MCS only
    #         assert isinstance(
    #             mcs_arr_eval_idx, int
    #         ), "Pre-defined MCS UE mask only works if mcs_arr_eval_idx is an integer"
    #         mcs_ue_mask = tf.one_hot(
    #             mcs_arr_eval_idx, depth=len(self._sys_parameters.mcs_index)
    #         )
    #         mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)
    #         mcs_ue_mask = tf.tile(
    #             mcs_ue_mask, multiples=[batch_size, self._sys_parameters.max_num_tx, 1]
    #         )
    #         mcs_arr_eval = [mcs_arr_eval_idx]
    #     else:
    #         # mcs_ue_mask is not none --> we now need to process all MCSs
    #         if isinstance(mcs_arr_eval_idx, (list, tuple)):
    #             # some different order specified. This is useful to evaluate
    #             # mixed MCS scenarios.
    #             # E2E model outputs data for mcs_arr_eval_idx[0], i.e., only
    #             # compute BER/BLER
    #             # for UEs scheduled with mcs_arr_eval_idx[0].
    #             assert len(mcs_arr_eval_idx) == len(
    #                 self._sys_parameters.mcs_index
    #             ), "mcs_arr_eval_idx list not compatible with length of mcs_index array"
    #             mcs_arr_eval = mcs_arr_eval_idx
    #         else:
    #             # process in order of mcs_index array
    #             mcs_arr_eval = list(range(len(self._sys_parameters.mcs_index)))

    #     ###################################
    #     # Transmitters
    #     # one transmitter for each supported MCS
    #     ###################################

    #     b = []
    #     for idx in range(len(mcs_arr_eval)):
    #         b.append(
    #             self._source(
    #                 [
    #                     batch_size,
    #                     self._sys_parameters.max_num_tx,
    #                     self._transmitters[mcs_arr_eval[idx]]._tb_size,
    #                 ]
    #             )
    #         )

    #     # Sample a random slot number and assigns its pilots to the transmitter
    #     if self._training:
    #         self._set_transmitter_random_pilots()

    #     # Combine transmit signals from all MCSs
    #     _mcs_ue_mask = tf.cast(
    #         expand_to_rank(
    #             tf.gather(mcs_ue_mask, indices=mcs_arr_eval[0], axis=2), 5, axis=-1
    #         ),
    #         dtype=tf.complex64,
    #     )
    #     x = _mcs_ue_mask * self._transmitters[mcs_arr_eval[0]](b[0])
    #     for idx in range(1, len(mcs_arr_eval)):
    #         _mcs_ue_mask = tf.cast(
    #             expand_to_rank(
    #                 tf.gather(mcs_ue_mask, indices=mcs_arr_eval[idx], axis=2),
    #                 5,
    #                 axis=-1,
    #             ),
    #             dtype=tf.complex64,
    #         )
    #         x = x + _mcs_ue_mask * self._transmitters[mcs_arr_eval[idx]](b[idx])

    #     # mask non-active DMRS ports by multiplying with 0
    #     a_tx = expand_to_rank(active_dmrs, tf.rank(x), axis=-1)
    #     x = tf.multiply(x, tf.cast(a_tx, tf.complex64))

    #     ###################################
    #     # Channel
    #     ###################################

    #     # Apply TX hardware impairments
    #     # CFO is applied per UE (i.e., must be done at TX side)
    #     if self._sys_parameters.frequency_offset is not None:
    #         x = self._sys_parameters.frequency_offset(x)

    #     # Rate adjusted SNR; for e2e learning non-rate adjusted is sometimes
    #     # preferred as pilotless communications changes the rate.
    #     if self._sys_parameters.ebno:

    #         # If pilot masking is used (for e2e), we account for the resulting
    #         # rate shift the assumption is that the empty REs are not
    #         # considered during transmission
    #         if self._sys_parameters.mask_pilots:
    #             tx = self._sys_parameters.transmitters[0]
    #             num_pilots = tf.cast(tx._resource_grid.num_pilot_symbols, tf.float32)
    #             num_res = tf.cast(tx._resource_grid.num_resource_elements, tf.float32)
    #             ebno_db -= 10.0 * log10(1.0 - num_pilots / num_res)
    #             # Remark: this also counts empty REs from oder CDM groups
    #             # (e.g., used for other DMRS ports).
    #             # In the current e2e config; this case does not occur.

    #         # Translate Eb/No [dB] to N0 for first evaluated MCS
    #         no = ebnodb2no(
    #             ebno_db,
    #             self._transmitters[mcs_arr_eval[0]]._num_bits_per_symbol,
    #             self._transmitters[mcs_arr_eval[0]]._target_coderate,
    #             self._transmitters[mcs_arr_eval[0]]._resource_grid,
    #         )

    #     else:
    #         # ebno_db is actually SNR when self._sys_parameters.ebno==False
    #         no = 10 ** (-ebno_db / 10)

    #     # Update topology only required for 3GPP UMi/UMa models
    #     if self._sys_parameters.channel_type in ("UMi", "UMa"):
    #         if self._sys_parameters.channel_type == "UMi":
    #             ch_type = "umi"
    #         else:
    #             ch_type = "uma"
    #         # Topology update only required for 3GPP pilot patterns
    #         topology = gen_single_sector_topology(
    #             batch_size,
    #             self._sys_parameters.max_num_tx,
    #             ch_type,
    #             min_ut_velocity=self._sys_parameters.min_ut_velocity,
    #             max_ut_velocity=self._sys_parameters.max_ut_velocity,
    #             indoor_probability=0.0,
    #         )  # disable indoor users
    #         self._sys_parameters.channel_model.set_topology(*topology)

    #     # Apply channel
    #     if self._sys_parameters.channel_type == "AWGN":
    #         y = self._channel([x, no])
    #         h = tf.ones_like(y)  # simple AWGN channel
    #     else:
    #         y, h = self._channel([x, no])

    #     ###################################
    #     # Receiver
    #     ###################################

    #     if self._sys_parameters.system in (
    #         "baseline_lmmse_kbest",
    #         "baseline_lmmse_lmmse",
    #         "baseline_lsnn_lmmse",
    #         "baseline_lslin_lmmse",
    #     ):
    #         b_hat = self._receiver([y, no])
    #         if self._return_tb_status:
    #             b_hat, tb_crc_status = b_hat
    #         else:
    #             tb_crc_status = None

    #         # return b[0] and b_hat only for active DMRS ports
    #         # b only holds bits corresponding to MCS indices specified
    #         # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
    #         return self._mask_active_dmrs(
    #             b[0], b_hat, num_tx, active_dmrs, mcs_arr_eval[0], tb_crc_status
    #         )

    #     elif self._sys_parameters.system in (
    #         "baseline_perf_csi_kbest",
    #         "baseline_perf_csi_lmmse",
    #     ):

    #         # perfect CSI receiver needs ground truth channel
    #         b_hat = self._receiver([y, h, no])

    #         if self._return_tb_status:
    #             b_hat, tb_crc_status = b_hat
    #         else:
    #             tb_crc_status = None
    #         # return b[0] and b_hat only for active DMRS ports
    #         # b only holds bits corresponding to MCS indices specified
    #         # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
    #         return self._mask_active_dmrs(
    #             b[0], b_hat, num_tx, active_dmrs, mcs_arr_eval[0], tb_crc_status
    #         )

    #     elif self._sys_parameters.system == "nrx":

    #         # in training mode, only the losses are required
    #         if self._training:
    #             losses = self._receiver(
    #                 [y, active_dmrs, b, h, mcs_ue_mask], mcs_arr_eval
    #             )
    #             return losses
    #         else:
    #             # in inference mode, the neural receiver returns:
    #             # - reconstructed payload bits b_hat
    #             # - refined channel estimate h_hat_refined
    #             # - initial channel estimate h_hat
    #             # - [optional] transport block CRC status
    #             b_hat, h_hat_refined, h_hat, tb_crc_status = self._receiver(
    #                 (y, active_dmrs), mcs_arr_eval, mcs_ue_mask_eval=mcs_ue_mask
    #             )

    #             #################################
    #             # Only focus on active DMRS ports
    #             #################################
    #             # Data
    #             # b only holds bits corresponding to MCS indices specified
    #             # in mcs_arr_eval --> evaluation for one MCS only --> b[0]
    #             b, b_hat, tb_crc_status = self._mask_active_dmrs(
    #                 b[0], b_hat, num_tx, active_dmrs, mcs_arr_eval[0], tb_crc_status
    #             )

    #             # Channel estimates
    #             h_hat_output_shape = tf.concat(
    #                 [[batch_size, num_tx], tf.shape(h_hat_refined)[2:]], axis=0
    #             )
    #             a_mask = expand_to_rank(active_dmrs, tf.rank(h_hat_refined), axis=-1)
    #             a_mask = tf.broadcast_to(a_mask, tf.shape(h_hat_refined))
    #             if h_hat is not None:
    #                 h_hat = tf.boolean_mask(h_hat, a_mask)
    #                 h_hat = tf.reshape(h_hat, h_hat_output_shape)
    #             h_hat_refined = tf.boolean_mask(h_hat_refined, a_mask)
    #             h_hat_refined = tf.reshape(h_hat_refined, h_hat_output_shape)
    #             # Channel ground truth
    #             h = self._receiver.preprocess_channel_ground_truth(h)
    #             h = tf.boolean_mask(h, a_mask)
    #             h = tf.reshape(h, h_hat_output_shape)

    #             # if activated, return channel estimates (and ground truth CFRs)
    #             if self._return_tb_status:
    #                 if output_nrx_h_hat:
    #                     return b, b_hat, tb_crc_status, h, h_hat_refined, h_hat
    #                 else:
    #                     return b, b_hat, tb_crc_status
    #             else:
    #                 if output_nrx_h_hat:
    #                     return b, b_hat, h, h_hat_refined, h_hat
    #                 else:
    #                     return b, b_hat
    #     else:
    #         raise ValueError("Unknown system selected!")


# class E2E_Model_Mixed_MCS(E2E_Model):
#     r"""E2E_Model_Mixed_MCS(sys_parameters, training=False, return_tb_status=False, **kwargs)
#     Wrapper for end-to-end model for system evaluation in mixed MCS scenarios.
#     This class allows to return b and b_hat and tb_crc_status only of one user.

#     The model can be used with the neural receiver as well as with the baseline
#     receivers.

#     For mixed MCS scenarios, you must provide mcs_arr_eval_idx as a list and
#     also specify an mcs_ue_mask. The receiver will only process and return bits
#     and bit estimates for mcs_arr_eval_idx[0].
#     Ensure that you select the right user index in ue_return that is scheduled
#     with mcs_arr_eval_idx[0].

#     Parameters
#     ----------
#     sys_parameters : Parameters
#         The system parameters.

#     training : bool
#         If True, the model returns the loss instead of b and b_hat.

#     return_tb_status : bool
#         If True, the model returns if the transport block CRC holds.

#     mcs_arr_eval_idx : int, list
#         Selects the element (index) of the mcs_index array to evaluate. Can be
#         overwritten as call argument.
#         Necessary to provide as a list for mixed MCS simulations. When
#         mcs_arr_eval_idx is a list,
#         it must have the same length as sys_parameters.mcs_index. Even when
#         provided as a list, the receiver
#         will only process mcs_arr_eval_idx[0]. Defaults to 0, which selects the
#         very first element specified in the mcs_index array.

#     ue_return : int
#         UE index to return the ground-truth bits and bit estimates. Defaults to
#         0.

#     mcs_ue_mask : None, or [batch_size, max_num_tx, len(mcs_index)], tf.int32
#         One-hot mask that specifies the MCS index of each UE for each batch
#         sample.
#         Must be specified when mcs_arr_eval_idx is provided as a list.
#         If not specified (defaults to None), the mcs_ue_mask will be inferred
#         from mcs_arr_eval_index.


#     Input
#     -----
#     batch_size : int
#         Batch size of random transmit signals to be generated

#     ebno_db: float
#         SNR in dB. Interpreted as rate-adjusted SNR if ``sys_parameters.ebno``
#         is `True`.

#     num_tx : int | None
#         Number of active DMRS ports. Remark: the model always simulates all
#         max_num_tx UEs. However, they are masked at the tx output, i.e., no
#         energy is transmitted. This allows training with varying number of
#         users without re-building the TensorFlow graph.

#     output_nrx_h_hat : bool
#         If True, the NRX internal channel estimations are returned.
#         This is required for training with double read-outs.


#     Output
#     ------
#     Depending on if return_tb_status and output_nrx_h_hat is True

#         Tuple : b, b_hat, [tb_crc_status], [h, h_hat_refined, h_hat]

#             b : [batch_size, num_tx, tb_size], tf.float
#                     Transmitted information bits for selected MCS.

#             b_hat : [batch_size, num_tx, tb_size], tf.float
#                 Decoded information bits for selected MCS.

#             tb_crc_status: [batch_size, num_tx], tf.float
#                 Status of the TB CRC for each decoded TB.

#             h: [batch_size, num_tx, num_effective_subcarriers,
#                     num_ofdm_symbols, 2*num_rx_ant]
#                 Ground truth channel CFRs

#             h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
#                     num_ofdm_symbols, 2*num_rx_ant]
#                 Refined channel estimate from the NRX.

#             h_hat, [batch_size, num_tx, num_effective_subcarriers,
#                    num_ofdm_symbols, 2*num_rx_ant]
#                 Initial channel estimate used for the NRX.
#     """

#     def __init__(
#         self,
#         sys_parameters,
#         training=False,
#         return_tb_status=False,
#         mcs_arr_eval_idx=0,
#         ue_return=0,
#         mcs_ue_mask=None,
#     ):
#         if isinstance(mcs_arr_eval_idx, (list, tuple)):
#             assert len(mcs_arr_eval_idx) == len(
#                 sys_parameters.mcs_index
#             ), "If mcs_arr_eval_idx is a list, it must have the same length as sys_parameters.mcs_index"
#             assert (
#                 mcs_ue_mask is not None
#             ), "Must specify mcs_ue_mask if mcs_arr_eval_idx is given as list"
#             super().__init__(
#                 sys_parameters=sys_parameters,
#                 training=training,
#                 return_tb_status=return_tb_status,
#                 mcs_arr_eval_idx=mcs_arr_eval_idx[0],
#             )
#             self._mcs_arr_eval = mcs_arr_eval_idx
#         else:
#             super().__init__(
#                 sys_parameters=sys_parameters,
#                 training=training,
#                 return_tb_status=return_tb_status,
#                 mcs_arr_eval_idx=mcs_arr_eval_idx,
#             )
#             self._mcs_arr_eval = mcs_arr_eval_idx
#         self._ue_return = ue_return

#         self._mcs_ue_mask = mcs_ue_mask

#     def call(self, batch_size, ebno_db, num_tx=None, output_nrx_h_hat=False):
#         if self._return_tb_status:
#             if output_nrx_h_hat:
#                 b, b_hat, tb_crc_status, h, h_hat_refined, h_hat = super().call(
#                     batch_size,
#                     ebno_db,
#                     num_tx,
#                     output_nrx_h_hat,
#                     mcs_arr_eval_idx=self._mcs_arr_eval,
#                     mcs_ue_mask=self._mcs_ue_mask,
#                 )
#             else:
#                 b, b_hat, tb_crc_status = super().call(
#                     batch_size,
#                     ebno_db,
#                     num_tx,
#                     output_nrx_h_hat,
#                     mcs_arr_eval_idx=self._mcs_arr_eval,
#                     mcs_ue_mask=self._mcs_ue_mask,
#                 )
#         else:
#             if output_nrx_h_hat:
#                 b, b_hat, h, h_hat_refined, h_hat = super().call(
#                     batch_size,
#                     ebno_db,
#                     num_tx,
#                     output_nrx_h_hat,
#                     mcs_arr_eval_idx=self._mcs_arr_eval,
#                     mcs_ue_mask=self._mcs_ue_mask,
#                 )
#             else:
#                 b, b_hat = super().call(
#                     batch_size,
#                     ebno_db,
#                     num_tx,
#                     output_nrx_h_hat,
#                     mcs_arr_eval_idx=self._mcs_arr_eval,
#                     mcs_ue_mask=self._mcs_ue_mask,
#                 )

#         b = tf.gather(b, indices=[self._ue_return], axis=1)
#         b_hat = tf.gather(b_hat, indices=[self._ue_return], axis=1)

#         if self._return_tb_status:
#             tb_crc_status = tf.gather(tb_crc_status, indices=[self._ue_return], axis=1)
#             if output_nrx_h_hat:
#                 return b, b_hat, tb_crc_status, h, h_hat_refined, h_hat
#             else:
#                 return b, b_hat, tb_crc_status
#         else:
#             if output_nrx_h_hat:
#                 return b, b_hat, h, h_hat_refined, h_hat
#             else:
#                 return b, b_hat
