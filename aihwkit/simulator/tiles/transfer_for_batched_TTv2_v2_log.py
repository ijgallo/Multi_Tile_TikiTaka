# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-instance-attributes

"""High level analog transfer tiles (analog)."""

from typing import Optional, Tuple, List, Dict, Any
from copy import deepcopy
from typing import Union

from math import ceil
import torch
from torch import Tensor, zeros, trunc, eye, rand, ones
from torch.nn import Module
from torch.autograd import no_grad
import math

from aihwkit.exceptions import ConfigError
from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery #AnalogTile
from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.multi_tile_periphery import MultiTileWithPeriphery
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.parameters.base import RPUConfigGeneric, RPUConfigBase
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound, DynamicTransferCompound
from aihwkit.simulator.tiles.analog_mvm import AnalogMVM


class BatchedTransferSimulatorTile(SimulatorTile, Module):
    """SimulatorTile for transfer.

    The RPUCuda library is only used for the single-tile forward / backward / pulsed
    update, however, not for the transfer from the gradient tile to the actual weight
    tile. The transfer part is implemented in python mostly for illustrative purposes and
    to allow for flexible adjustments and development of new algorithms based on the
    Tiki-taka approach.

    Note:
        Only a subset of parameter settings are supported.

    Caution:

        The construction seed that is applied for both tiles when using
        :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound` is here not
        applied, unless given explicitly for the unit cell devices.

    Args:
        out_size: output size
        in_size: input size
        rpu_config: resistive processing unit configuration.
        dtype: data type to use for the tiles.

    Raises:
        ConfigError: in case a setting is not supported.

    """

    def __init__(
        self, x_size: int, d_size: int, rpu_config: "UnitCellRPUConfig", dtype: RPUDataType
    ):
        Module.__init__(self)

        self.x_size = x_size
        self.d_size = d_size
        self.update_counter = 0
        self.chop_counter = 0
        self.transfer_idx = 0
        self.learning_rate = 0.1
        self.rpu_config = rpu_config

        self.m_x = 1.0
        self.m_d = 1.0

        self._transfer_vec = None  # type: Optional[Tensor]

        if not isinstance(rpu_config.device, ChoppedTransferCompound):
            raise ConfigError("Device chould be of type Chopped/Dynamic TransferCompound")
        
        if hasattr(rpu_config, "batch_size"):
            self.num_tiles = rpu_config.batch_size
        else:
            rpu_config.batch_size = 1
            self.num_tiles = 1

        self.device_config = deepcopy(rpu_config.device)

        cfg = self.device_config
        rpu_config_0 = SingleRPUConfig(
            mapping=deepcopy(rpu_config.mapping),
            pre_post=deepcopy(rpu_config.pre_post),
            device=deepcopy(cfg.unit_cell_devices[0]),
            forward=deepcopy(cfg.transfer_forward),
            backward=deepcopy(cfg.transfer_forward),
            update=deepcopy(rpu_config.update),
            tile_class=AnalogTileWithoutPeriphery,
        )
        rpu_config_1 = SingleRPUConfig(
            mapping=deepcopy(rpu_config.mapping),
            pre_post=deepcopy(rpu_config.pre_post),
            device=deepcopy(cfg.unit_cell_devices[1]),
            forward=deepcopy(rpu_config.forward),
            backward=deepcopy(rpu_config.backward),
            update=deepcopy(cfg.transfer_update),
            tile_class=AnalogTileWithoutPeriphery,
        )

        self.grad_tiles = [rpu_config_0.tile_class(d_size, x_size, rpu_config_0) for _ in range(self.num_tiles)]
        self.weight_tiles = [rpu_config_1.tile_class(d_size, x_size, rpu_config_1) for _ in range(self.num_tiles)]

        self.from_weight_granularity = rpu_config_0.device.as_bindings(
            dtype
        ).calc_weight_granularity()
        self.to_weight_granularity = rpu_config_1.device.as_bindings(
            dtype
        ).calc_weight_granularity()

        lr_w = self.device_config.step * self.to_weight_granularity
        [weight_tile.set_learning_rate(lr_w) for weight_tile in self.weight_tiles]

        transfer_columns = self.device_config.transfer_columns
        self.t_in_size = self.x_size if transfer_columns else self.d_size
        self.t_out_size = self.d_size if transfer_columns else self.x_size

        hidden_weight = zeros([self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
        self.register_buffer("hidden_weight", hidden_weight)

        if isinstance(cfg, DynamicTransferCompound):
            past_mean_weight = zeros([self.num_tiles, self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
            self.register_buffer("past_mean_weight", past_mean_weight)
            reference_weight = zeros([self.num_tiles, self.t_in_size, self.t_out_size], dtype=dtype.as_torch())
            self.register_buffer("reference_weight", reference_weight)

        chopper = ones([self.x_size], dtype=dtype.as_torch())
        self.register_buffer("chopper", chopper)

        # set auto-scale base scale
        granularity = rpu_config.update.desired_bl * self.from_weight_granularity
        self.lr_a_auto_scale = cfg.fast_lr * granularity
        self.desired_bl = rpu_config.update.desired_bl
        self.transfer_bl = rpu_config.device.transfer_update.desired_bl
        self.forward_noise = rpu_config_1.forward.out_noise

        self.log = []
        self.dlog = []

    # Define all possible keys for complete dataframe compatibility
    def create_log_entry(self, operation_type, operation, meta_operation, complexity, energy_profile, 
                     vector_length=None, times=1, notes=None):
        return {
            "Type": operation_type,        # "digital" or "analog"
            "Op": operation,               # Specific operation name
            "Meta_op": meta_operation,          # Meta operation name
            "Complexity": complexity,      # Big O notation
            "Energy_profile": energy_profile,  # relative energy cost
            "Vector_length": vector_length,    # for vector operations
            "Times": times,                # repetition count
            "Notes": notes                 # optional context
        }


    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """
        Distribute input across multiple weight tiles and process in parallel,
        allowing multiple inputs per tile if batch size exceeds available tiles.
        
        Args:
            x_input: Input tensor
            bias: Whether to use bias
            in_trans: Input transformation flag
            out_trans: Output transformation flag
            is_test: Test mode flag
            non_blocking: Non-blocking computation flag
        
        Returns:
            Tensor with outputs from all tiles
        """
        in_batch_size = x_input.size(0)
        inputs_per_tile = (in_batch_size + self.num_tiles - 1) // self.num_tiles
        
        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "forward", "O(1)", "very_low", times=3
        ))
        
        outputs = torch.stack([
            weight_tile.tile.forward(
                x_input[i:i + inputs_per_tile],
                bias,
                in_trans,
                out_trans,
                is_test,
                non_blocking
            )
            for i, weight_tile in zip(
                range(0, in_batch_size, inputs_per_tile),
                self.weight_tiles
            )
        ])

        self.log.append({
                        "Type": "analog",
                        "Tile": "main",
                        "Op": "forward",
                        "Meta_op": "forward",
                        "Times": in_batch_size,
                        "Pfactor": in_batch_size / inputs_per_tile,
                        })
        
        # for each element in the batch
        self.dlog.append(self.create_log_entry(
            "digital_la", "abs", "forward", "O(n)", "low",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "max", "forward", "O(n)", "low",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))
        # scaling down
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))
        # rescaling
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))

        # buffers
        # buffer for the inputs
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))

        # buffer for intermidiate results
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-read", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
        ))

        # bias
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-read", "forward", "O(n)", "medium",
            vector_length=outputs.shape[-1], times=in_batch_size
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-add", "forward", "O(n)", "medium",
            vector_length=outputs.shape[-1], times=in_batch_size
        ))

        # activation function
        self.dlog.append(self.create_log_entry(
            "digital_la", "activation", "forward", "O(n)", "medium",
            vector_length=outputs.shape[-1], times=in_batch_size
        ))

        bm_rounds = AnalogMVM.matmul(self.weight_tiles[0].tile.get_weights(), x_input, self.rpu_config.forward)
        for _ in range(1, bm_rounds):
            self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "forward",
                    "Meta_op": "forward",
                    "Times": in_batch_size,
                    "Pfactor": in_batch_size / inputs_per_tile,
                    })
            # scaling down
            self.dlog.append(self.create_log_entry(
                "digital_la", "vector-element", "forward", "O(n)", "medium",
                vector_length=x_input.shape[-1], times=in_batch_size
            ))

            self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "forward", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=in_batch_size
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "buffer-read", "forward", "O(n)", "medium",
                vector_length=x_input.shape[-1], times=in_batch_size
            ))

        return outputs.reshape(in_batch_size, -1)

    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """
        Backward pass handling multiple inputs per tile if batch size exceeds available tiles.
        Only needs to be implemented if torch autograd is not used.
        """
        d_batch_size = d_input.size(0)
        inputs_per_tile = (d_batch_size + self.num_tiles - 1) // self.num_tiles

        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "backward", "O(1)", "very_low", times=3
        ))
        
        outputs = torch.stack([
            weight_tile.tile.backward(
                d_input[i:i + inputs_per_tile],
                bias,
                in_trans,
                out_trans,
                non_blocking
            )
            for i, weight_tile in zip(
                range(0, d_batch_size, inputs_per_tile),
                self.weight_tiles
            )
        ])

        self.log.append({
                        "Type": "analog",
                        "Tile": "main",
                        "Op": "backward",
                        "Meta_op": "backward",
                        "Times": d_batch_size,
                        "Pfactor": d_batch_size / inputs_per_tile,
                        })
        
        # for each element in the batch
        self.dlog.append(self.create_log_entry(
            "digital_la", "abs", "backward", "O(n)", "low",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "max", "backward", "O(n)", "low",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))
        # scaling down
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))
        # rescaling
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))

        # buffers
        # buffer for the inputs
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))

        # buffer for intermidiate results (bound management)
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-read", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
        ))

        # activation function
        self.dlog.append(self.create_log_entry(
            "digital_la", "act-derivative", "backward", "O(n)", "medium",
            vector_length=outputs.shape[-1], times=d_batch_size
        ))

        bm_rounds = AnalogMVM.matmul(self.weight_tiles[0].tile.get_weights().T, d_input, self.rpu_config.forward)
        for _ in range(1, bm_rounds):
            self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "backward",
                    "Meta_op": "backward",
                    "Times": d_batch_size,
                    "Pfactor": d_batch_size / inputs_per_tile,
                    })
            # scaling down
            self.dlog.append(self.create_log_entry(
                "digital_la", "vector-element", "backward", "O(n)", "medium",
                vector_length=d_input.shape[-1], times=d_batch_size
            ))

            self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-write", "backward", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=d_batch_size
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "buffer-read", "backward", "O(n)", "medium",
                vector_length=d_input.shape[-1], times=d_batch_size
            ))
        
        return outputs.reshape(d_batch_size, -1)

    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Transfer update."""

        # pylint: disable=too-many-branches, too-many-statements
        # pylint: disable=attribute-defined-outside-init, too-many-locals

        cfg = self.device_config
        if in_trans or out_trans or bias:
            raise ConfigError("Trans or bias not supported ")
        if not cfg.n_reads_per_transfer == 1:
            raise ConfigError("Only 1 read per transfer supported")
        # just assume m batch for now
        if cfg.transfer_every < 0:
            raise ConfigError("Auto transfer every not supported")

        m_batch = 1
        if x_input.dim() > 1:
            m_batch = x_input.size(int(in_trans))

        transfer_every = cfg.transfer_every
        if cfg.units_in_mbatch:
            transfer_every = m_batch * transfer_every
        transfer_every = int(ceil(transfer_every))

        if isinstance(cfg, DynamicTransferCompound) and cfg.experimental_correct_accumulation:
            # also not supported is buffer_cap
            raise ConfigError("Correct accumulation is not supported")

        # dynamically adjust learning rates

        if cfg.auto_scale:
            lr_a = self.lr_a_auto_scale
            tau = (1.0 - cfg.auto_momentum) / m_batch

            m_x = x_input.abs().max().item()
            m_d = d_input.abs().max().item()

            self.m_x = (1 - tau) * self.m_x + tau * m_x  # type: ignore
            self.m_d = (1 - tau) * self.m_d + tau * m_d  # type: ignore

            self.dlog.append(self.create_log_entry(
                "digital_la", "abs", "update", "O(n)", "low", 
                vector_length=x_input.shape[-1], times=x_input.shape[0]
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "abs", "update", "O(n)", "low", 
                vector_length=d_input.shape[-1], times=d_input.shape[0]
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "max", "update", "O(n)", "medium", 
                vector_length=x_input.shape[-1], times=x_input.shape[0]
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "max", "update", "O(n)", "medium", 
                vector_length=d_input.shape[-1], times=d_input.shape[0]
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "arithmetic", "update", "O(1)", "very_low", times=8
            ))

            if self.m_x > 0.0 and self.m_d > 0.0:
                lr_a /= self.m_x * self.m_d
                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "update", "O(1)", "very_low", times=1
                ))
            
            self.dlog.append(self.create_log_entry(
                "digital_la", "conditional", "update", "O(1)", "very_low", times=1
            ))

        else:
            lr_a = cfg.fast_lr
            self.dlog.append(self.create_log_entry(
                "digital_la", "getter_setter", "update", "O(1)", "very_low", times=1
            ))


        # gradient accumulation
        [grad_tile.set_learning_rate(lr_a) for grad_tile in self.grad_tiles]

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "update", "O(1)", "very_low", times=3
        ))

        if self.chopper is not None:
            x_input *= self.chopper
            self.dlog.append(self.create_log_entry(
                "digital_la", "vector-element", "update", "O(n)", "very_low",
                vector_length=x_input.numel(), times=1,
                notes="Chopper application"
            ))

        # access buffers
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-read", "update", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=m_batch,
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "buffer-read", "update", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=m_batch,
        ))

        # generate pulse train
        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "update", "O(n)", "medium",
            vector_length=x_input.shape[-1], times=m_batch,
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "update", "O(n)", "medium",
            vector_length=d_input.shape[-1], times=m_batch,
        ))

        in_batch_size = x_input.size(0)
        inputs_per_tile = (in_batch_size + self.num_tiles - 1) // self.num_tiles
        for i, grad_tile in zip(range(0, in_batch_size, inputs_per_tile), self.grad_tiles):
            grad_tile.tile.update(x_input[i:i + inputs_per_tile], d_input[i:i + inputs_per_tile], bias, in_trans, out_trans, non_blocking)
                
        self.update_counter += m_batch

        self.log.append({
                        "Type": "analog",
                        "Tile": "aux",
                        "Op": "update",
                        "Meta_op": "update",
                        "Times": m_batch,
                        "Pfactor": m_batch / inputs_per_tile,
                        })

        # handle transfer (Note that it will only update once for the m_batch!)
        rest_count = ((self.update_counter - m_batch) % transfer_every) + m_batch

        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "update", "O(1)", "very_low", times=7
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "conditional", "update", "O(1)", "very_low", times=4
        ))

        if rest_count >= transfer_every:
            if rest_count >= 2 * transfer_every:
                raise ConfigError(
                    "Multiple transfers within batch not supported"
                    " for the python transfer implementation."
                )

            # transfer learning rate lr_h
            buffer_granularity = cfg.buffer_granularity if cfg.buffer_granularity > 0.0 else 1.0
            if cfg.auto_granularity > 0.0:
                period = self.t_in_size * transfer_every
                buffer_granularity *= self.from_weight_granularity * cfg.auto_granularity / period
                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=4
                ))
            else:
                buffer_granularity *= self.from_weight_granularity
                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=1
                ))

            if cfg.correct_gradient_magnitudes:
                buffer_granularity *= self.to_weight_granularity / self.from_weight_granularity
                lr_h = self.learning_rate / lr_a / buffer_granularity
                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=4
                ))
            else:
                lr_h = self.learning_rate / buffer_granularity
                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=1
                ))

            chop_period = int(round(1.0 / cfg.in_chop_prob)) if cfg.in_chop_prob > 0 else 1

            if self._transfer_vec is None:
                # construct on the fly
                self._transfer_vec = eye(self.t_in_size, dtype=x_input.dtype, device=x_input.device)
                self.dlog.append(self.create_log_entry(
                    "digital_la", "eye", "transfer", "O(n²)", "high", 
                    vector_length=self.t_in_size, times=1,
                    notes="Identity matrix creation"
                ))

            self.transfer_idx = (self.transfer_idx + 1) % self.t_in_size

            k = self.transfer_idx

            # read gradients
            if cfg.transfer_columns:
                omega = [grad_tile.tile.forward(
                    self._transfer_vec[k], False, in_trans, out_trans, False, non_blocking
                ) for grad_tile in self.grad_tiles]
            else:
                omega = [grad_tile.tile.backward(
                    self._transfer_vec[k], False, in_trans, out_trans, non_blocking
                ) for grad_tile in self.grad_tiles]
            
            self.log.append({
                            "Type": "analog",
                            "Tile": "aux",
                            "Op": "forward",
                            "Meta_op": "transfer",
                            "Times": self.num_tiles,
                            "Pfactor": self.num_tiles,
                        })

            omega = torch.stack(omega)

            # getting chopper
            self.dlog.append(self.create_log_entry(
                "digital_la", "buffer-read", "transfer", "O(1)", "medium",
                vector_length=1, times=1,
            ))

            # update hidden buffer
            if isinstance(cfg, DynamicTransferCompound):
                beta = min(cfg.tail_weightening / chop_period, 1)
                self.past_mean_weight[:, k] = (1 - beta) * self.past_mean_weight[:, k] + beta * omega
                diff = torch.sum(omega - self.reference_weight[:, k], dim=0)
                self.hidden_weight[k] += lr_h * self.chopper[k] * diff

                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=3
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-element", "transfer", "O(n)", "medium",
                    vector_length=omega.shape[-1], times=2 * self.num_tiles + 1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-mult", "transfer", "O(n)", "medium",
                    vector_length=omega.shape[-1], times=1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-add", "transfer", "O(n)", "medium",
                    vector_length=omega.shape[-1], times=2 * self.num_tiles + 1
                ))

                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-add", "transfer", "O(n)", "medium",
                    vector_length=omega.shape[-1], times=self.num_tiles
                ))   

                # buffers
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-read", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=2*self.num_tiles
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-write", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=2*self.num_tiles
                ))
            else:
                self.hidden_weight[k] += lr_h * self.chopper[k] * omega
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-element", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-mult", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "vector-add", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=1
                ))

                # buffers
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-read", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-write", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=1
                ))


            # compute update weight
            write_values = -trunc(self.hidden_weight[k])  # negative because of update LR
            if cfg.forget_buffer:
                self.hidden_weight[k][write_values != 0] = cfg.momentum
            else:
                self.hidden_weight[k] += write_values * (1 - cfg.momentum)

            self.dlog.append(self.create_log_entry(
                "digital_la", "vector-element", "transfer", "O(n)", "medium",
                vector_length=omega.numel(), times=2
            ))

            # handle chopper
            switch_chopper = False
            if cfg.in_chop_prob:
                if cfg.in_chop_random:
                    switch_chopper = rand(1).item() < cfg.in_chop_prob

                    self.dlog.append(self.create_log_entry(
                        "digital_la", "randn", "transfer", "O(1)", "low", times=1
                    ))
                    self.dlog.append(self.create_log_entry(
                        "digital_la", "comparison", "transfer", "O(1)", "very_low", times=1
                    ))
                else:
                    if k == 0:
                        self.chop_counter = (self.chop_counter + 1) % chop_period
                        self.dlog.append(self.create_log_entry(
                            "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=2
                        ))
                    switch_chopper = self.chop_counter == 0
                    self.dlog.append(self.create_log_entry(
                        "digital_la", "conditional", "transfer", "O(1)", "very_low", times=2
                    ))
                    self.dlog.append(self.create_log_entry(
                        "digital_la", "comparison", "transfer", "O(1)", "very_low", times=1
                    ))

            if switch_chopper:
                self.chopper[k] = -self.chopper[k]

                self.dlog.append(self.create_log_entry(
                    "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=1
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-write", "transfer", "O(1)", "medium",
                    vector_length=1, times=1
                ))

            self.dlog.append(self.create_log_entry(
                "digital_la", "abs", "transfer", "O(n)", "low",
                vector_length=write_values.numel(), times=1
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "sum", "transfer", "O(n)", "medium",
                vector_length=write_values.numel(), times=1
            ))
            self.dlog.append(self.create_log_entry(
                "digital_la", "comparison", "transfer", "O(1)", "very_low", times=1
            ))
            # write to weight
            if not write_values.abs().sum() == 0.0:
                for _, weight_tile in enumerate(self.weight_tiles):
                    if cfg.transfer_columns:
                        weight_tile.tile.update(
                            self._transfer_vec[k],
                            write_values,
                            False,
                            in_trans,
                            out_trans,
                            non_blocking,
                        )
                    else:
                        weight_tile.tile.update(
                            write_values,
                            self._transfer_vec[k],
                            False,
                            in_trans,
                            out_trans,
                            non_blocking,
                        )

                self.log.append({
                            "Type": "analog",
                            "Tile": "main",
                            "Op": "update",
                            "Meta_op": "transfer",
                            "Times": self.num_tiles,
                            "Pfactor": self.num_tiles,
                        })

            # additional compute for AGAD
            if isinstance(cfg, DynamicTransferCompound) and switch_chopper:
                self.reference_weight[:, k] = self.past_mean_weight[:, k]
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-read", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=self.num_tiles
                ))
                self.dlog.append(self.create_log_entry(
                    "digital_la", "buffer-write", "transfer", "O(n)", "medium",
                    vector_length=omega.numel(), times=self.num_tiles
                ))

            self.dlog.append(self.create_log_entry(
                "digital_la", "conditional", "transfer", "O(1)", "very_low", times=13
            ))

            self.dlog.append(self.create_log_entry(
                "digital_la", "arithmetic", "transfer", "O(1)", "very_low", times=5
            ))

            self.dlog.append(self.create_log_entry(
                "digital_la", "getter_setter", "transfer", "O(1)", "very_low", times=1
            ))

    def get_brief_info(self) -> str:
        """Returns a brief info"""
        return self.__class__.__name__ + "({})".format(self.extra_repr())

    def get_weights(self) -> List[Tensor]:
        """Returns the analog weights."""

        return torch.stack([weight_tile.tile.get_weights() for weight_tile in self.weight_tiles])


    @no_grad()
    def read_weights_multi(
        self,
        x_values: Optional[torch.Tensor] = None,
        over_sampling: int = 10
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Read weights from multiple tiles using least squares estimation.
        
        Args:
            x_values: Input values for weight estimation. 
                    If None, generate random inputs.
            over_sampling: Number of additional samples to use 
                        for more robust estimation
        
        Returns:
            List of tuples, each containing:
            - Estimated weight matrix for a tile
            - Optional bias for that tile
        """        
        # Generate random inputs if not provided
        if x_values is None:
            x_values = torch.randn(
                self.x_size * self.num_tiles, 
                self.x_size, 
            )
            self.dlog.append(self.create_log_entry(
                "digital_la", "randn", "read", "O(n)", "high", 
                vector_length=self.x_size, times=self.num_tiles*self.x_size
            ))
        
        # Ensure the object is a valid module
        if not isinstance(self, torch.nn.Module):
            raise ValueError("Must be a neural network module")
        
        # Temporarily switch to evaluation mode
        was_training = self.training
        self.eval()

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "read", "O(1)", "very_low", times=2
        ))
        
        # Forward pass for this tile
        y_values = self.forward(x_values)
        # Solve least squares problem
        solution = torch.linalg.lstsq(x_values, y_values).solution
        est_weight = solution.T.cpu()

        self.dlog.append(self.create_log_entry(
            "digital_la", "lstsq", "read", "O(n³)", "very_high", 
            vector_length=self.x_size, times=1,
            notes="Solving least squares system"
        ))
        
        self.dlog.append(self.create_log_entry(
            "digital_la", "transpose", "read", "O(n²)", "medium", 
            vector_length=self.x_size, times=1,
        ))
    
        if was_training:
            self.train()

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "read", "O(1)", "very_low", times=self.num_tiles + 1
        ))

        # scaling 
        self.dlog.append(self.create_log_entry(
            "digital_pe", "vector-element", "read", "O(n)", "medium",
            vector_length=self.x_size, times=self.d_size
        ))
        
        self.log.append({
                        "Type": "analog",
                        "Tile": "main",
                        "Op": "forward",
                        "Meta_op": "read",
                        "Times": self.x_size * self.num_tiles,
                        "Pfactor": self.num_tiles,
                        })

        return est_weight
    
    @no_grad()
    def read_weights(
        self,
        x_values: Optional[torch.Tensor] = None,
        over_sampling: int = 10
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Read weights from multiple tiles using least squares estimation.
        
        Args:
            x_values: Input values for weight estimation. 
                    If None, generate random inputs.
            over_sampling: Number of additional samples to use 
                        for more robust estimation
        
        Returns:
            List of tuples, each containing:
            - Estimated weight matrix for a tile
            - Optional bias for that tile
        """        
        # Generate random inputs if not provided
        if x_values is None:
            x_values = torch.randn(
                self.x_size * over_sampling, 
                self.x_size, 
            )
            self.dlog.append(self.create_log_entry(
                "digital_la", "randn", "read", "O(n)", "high", 
                vector_length=self.x_size, times=over_sampling*self.x_size
            ))
        
        # Ensure the object is a valid module
        if not isinstance(self, torch.nn.Module):
            raise ValueError("Must be a neural network module")
        
        # Temporarily switch to evaluation mode
        was_training = self.training
        self.eval()

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "read", "O(1)", "very_low", times=2
        ))
        
        tile_weights = []
        # Estimate weights for each tile
        for weight_tile in self.weight_tiles:
            # Forward pass for this tile
            y_values = weight_tile.tile.forward(x_values)
            # Solve least squares problem
            solution = torch.linalg.lstsq(x_values, y_values).solution
            est_weight = solution.T.cpu()
            tile_weights.append(est_weight)

        self.dlog.append(self.create_log_entry(
            "digital_la", "lstsq", "read", "O(n³)", "very_high", 
            vector_length=self.x_size, times=self.num_tiles,
            notes="Solving least squares system"
        ))
        
        self.dlog.append(self.create_log_entry(
            "digital_la", "transpose", "read", "O(n²)", "medium", 
            vector_length=self.x_size, times=self.num_tiles,
        ))
        
        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "read", "O(1)", "very_low", times=self.num_tiles + 1
        ))
    
        if was_training:
            self.train()

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "read", "O(1)", "very_low", times=self.num_tiles + 1
        ))
        
        self.log.append({
                        "Type": "analog",
                        "Tile": "main",
                        "Op": "forward",
                        "Meta_op": "read",
                        "Times": self.x_size * over_sampling * self.num_tiles,
                        "Pfactor": self.num_tiles,
                        })

        return torch.stack(tile_weights)
    

    def set_weights(self, weights: Tensor) -> None:
        """Stets the analog weights."""
        for weight_tile, weight in zip(self.weight_tiles, weights):
            weight_tile.tile.set_weights(weight)

    @no_grad()
    def program_weights(
        self,
        weights: Tensor,
        x_values: Optional[Tensor] = None,
        learning_rate: float = 0.25,
        max_iter: int = 10000,
        tolerance: Optional[float] = 0.02,
        w_init: Union[float, Tensor] = 0.4,
    ) -> None:
        """Programm the target weights into the conductances using the
        pulse update defined.

        Programming is done using the defined tile-update (e.g. SGD)
        and matching inputs (`x_values` by default `eye`).

        Args:

            from_reference: Whether to use weights from reference
                (those that were initally set with `set_weights`) or
                the current weights.
            x_values: Values to use for the read-and verify. If none
                are given, unit-vectors are used
            learning_rate: Learning rate of the optimization
            max_iter: max number of batches for the iterative programming
            tolerance: Stop the iteration loop early if the mean
                output deviation is below this number. Given in
                relation to the max output.
            w_init: initial weight matrix to start from. If given as
                float, weights are set uniform random in `[-w_init,
                w_init]`. This init weight is given directly in
                normalized conductance units and should include the
                bias row if existing.
        """

        iters_used = []
        learning_rate = learning_rate / self.transfer_bl
        for weight_tile, target_weights in zip(self.weight_tiles, weights):
            if x_values is None:
                x_values = eye(self.x_size)
                target_values = x_values @ target_weights.T

            std_threshold = math.sqrt(tolerance**2 + 0.1**2 + 5e-4)  # Adjusted threshold

            if isinstance(w_init, Tensor):
                weight_tile.tile.set_weights(w_init)
            else:
                weight_tile.tile.set_weights(torch.randn_like(weight_tile.tile.get_weights()) * w_init)
            
            lr_save = weight_tile.tile.get_learning_rate()  # type: ignore
            weight_tile.tile.set_learning_rate(learning_rate)  # type: ignore
            
            counter = 0
            running_mean = std_threshold
            for _ in range(max_iter):
                y = weight_tile.tile.forward(x_values)
                error = y - target_values
                running_mean = (1 - 0.1) * running_mean + 0.1 * torch.std(error).item()
                if tolerance is not None and running_mean < std_threshold:
                    break
                weight_tile.update(x_values, error)  # type: ignore
                counter += 1
            iters_used.append(counter)

            weight_tile.tile.set_learning_rate(lr_save)  # type: ignore

        avg_iter = sum(iters_used) / len(iters_used)

        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "forward",
                    "Meta_op": "program",
                    "Times": avg_iter * self.x_size * self.num_tiles,
                    "Pfactor": self.num_tiles,
                    })
        
        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "update",
                    "Meta_op": "program",
                    "Times": avg_iter * self.x_size * self.num_tiles,
                    "Pfactor": self.num_tiles,
                    })
        
    @no_grad()
    def program_weights_from_zero(
        self,
        weights: Tensor,
        x_values: Optional[Tensor] = None,
        learning_rate: float = 0.25,
        max_iter: int = 250,
        tolerance: Optional[float] = 0.025,
        w_init: Union[float, Tensor] = 0.4,
        max_iter_zero: int = 2500,
    ) -> None:
        """Programm the target weights into the conductances using the
        pulse update defined.

        Programming is done using the defined tile-update (e.g. SGD)
        and matching inputs (`x_values` by default `eye`).

        Args:

            from_reference: Whether to use weights from reference
                (those that were initally set with `set_weights`) or
                the current weights.
            x_values: Values to use for the read-and verify. If none
                are given, unit-vectors are used
            learning_rate: Learning rate of the optimization
            max_iter: max number of batches for the iterative programming
            tolerance: Stop the iteration loop early if the mean
                output deviation is below this number. Given in
                relation to the max output.
            w_init: initial weight matrix to start from. If given as
                float, weights are set uniform random in `[-w_init,
                w_init]`. This init weight is given directly in
                normalized conductance units and should include the
                bias row if existing.
        """

        iters_used = []
        learning_rate = learning_rate / self.transfer_bl
        for weight_tile, target_weights in zip(self.weight_tiles, weights):
            
            if x_values is None:
                x_values = eye(self.x_size)
                target_values = x_values @ target_weights.T

            if isinstance(w_init, Tensor):
                weight_tile.tile.set_weights(w_init)
            else:
                weight_tile.tile.set_weights(torch.randn_like(weight_tile.tile.get_weights()) * w_init)
            
            lr_save = weight_tile.tile.get_learning_rate()  # type: ignore
            weight_tile.tile.set_learning_rate(learning_rate)  # type: ignore
            
            # firstly, train them to zero
            for _ in range(max_iter_zero):
                x_in = torch.randn(self.x_size)
                y = weight_tile.tile.forward(x_in)
                error = y
                weight_tile.update(x_in, error)  # type: ignore

            variance_threshold = tolerance**2 + (self.forward_noise + 2.2e-3*(0.1/self.forward_noise))**2 if self.forward_noise > 0 else tolerance**2
            alpha = 0.1  # Smoothing factor
            ewmv = variance_threshold * 2  # Initialize above threshold
            counter = 0
            for _ in range(max_iter):
                y = weight_tile.tile.forward(x_values)
                error = y - target_values
                # Calculate the squared error for this iteration
                current_var = torch.var(error).item()
                # Update the exponentially weighted moving variance
                ewmv = (1 - alpha) * ewmv + alpha * current_var
                if ewmv < variance_threshold:
                    break
                weight_tile.update(x_values, error)  # type: ignore
                counter += 1
            iters_used.append(counter)

            weight_tile.tile.set_learning_rate(lr_save)  # type: ignore

        total_iter = sum(iters_used)
        max_iters = max(iters_used)
        print("Average iterations:", total_iter / len(iters_used))

        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "forward",
                    "Meta_op": "program",
                    "Times": total_iter * self.x_size + max_iter_zero * self.num_tiles,
                    "Pfactor": total_iter/max_iters,
                    })
        
        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "update",
                    "Meta_op": "program",
                    "Times": total_iter * self.x_size + max_iter_zero * self.num_tiles,
                    "Pfactor": total_iter/max_iters,
                    })
        
    @no_grad()
    def program_weights_from_one(
        self,
        weights: Tensor,
        learning_rate: float = 0.25,
        max_iter: int = 2500,
        tolerance: Optional[float] = 0.025,
        w_init: Union[float, Tensor] = 0.4,
        max_iter_zero: int = 2500,
    ) -> None:
        """Programm the target weights into the conductances using the
        pulse update defined.

        Programming is done using the defined tile-update (e.g. SGD)
        and matching inputs (`x_values` by default `eye`).

        Args:

            from_reference: Whether to use weights from reference
                (those that were initally set with `set_weights`) or
                the current weights.
            x_values: Values to use for the read-and verify. If none
                are given, unit-vectors are used
            learning_rate: Learning rate of the optimization
            max_iter: max number of batches for the iterative programming
            tolerance: Stop the iteration loop early if the mean
                output deviation is below this number. Given in
                relation to the max output.
            w_init: initial weight matrix to start from. If given as
                float, weights are set uniform random in `[-w_init,
                w_init]`. This init weight is given directly in
                normalized conductance units and should include the
                bias row if existing.
        """

        learning_rate = learning_rate / self.transfer_bl

        # set initial conditions
        for weight_tile in self.weight_tiles:
            if isinstance(w_init, Tensor):
                weight_tile.tile.set_weights(w_init)
            else:
                weight_tile.tile.set_weights(torch.randn_like(weight_tile.tile.get_weights()) * w_init)

        # get tile 0 close to the std of the target weights
        tile_0 = self.weight_tiles[0]
        target_std = torch.std(weights[0]).item()
        lr_save = tile_0.tile.get_learning_rate()  # type: ignore
        tile_0.tile.set_learning_rate(learning_rate)  # type: ignore

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "program", "O(1)", "very_low", times=2
        ))

        norm_factor = math.sqrt(self.x_size)
        abs_max = (2*math.log(self.x_size) - math.log(math.log(self.x_size)))/self.x_size
        target_var = target_std**2
        noise_var = (self.forward_noise + 2.2e-3*(0.1/self.forward_noise))**2 * abs_max if self.forward_noise > 0 else 0
        variance_threshold = target_var + noise_var
        alpha = 0.05  # Smoothing factor
        ewmv = variance_threshold * 2  # Initialize above threshold

        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "program", "O(1)", "very_low", times=10
        ))

        iters_used_0 = 0
        for _ in range(max_iter_zero):
            x_in = torch.randn(self.x_size)
            x_in = x_in / norm_factor
            y = tile_0.tile.forward(x_in)
            error = y
            current_var = torch.var(error).item()
            ewmv = (1 - alpha) * ewmv + alpha * current_var
            if ewmv < variance_threshold:
                break
            tile_0.update(x_in, error)  # type: ignore
            iters_used_0 += 1

        self.dlog.append(self.create_log_entry(
            "digital_la", "loop_overhead", "program", "O(1)", "low", times=iters_used_0
        ))
        # Random vector generation
        self.dlog.append(self.create_log_entry(
            "digital_la", "randn", "program", "O(n)", "high", vector_length=self.x_size, times=iters_used_0
        ))

        # Vector normalization
        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "program", "O(n)", "medium", vector_length=self.x_size, times=iters_used_0
        ))

        # Variance calculation
        self.dlog.append(self.create_log_entry(
            "digital_la", "variance", "program", "O(n)", "medium", vector_length=y.numel(), times=iters_used_0
        ))

        # EWMA calculation and count increment
        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "program", "O(1)", "very_low", times=iters_used_0 * 4
        ))

        # Conditional check
        self.dlog.append(self.create_log_entry(
            "digital_la", "comparison", "program", "O(1)", "very_low", times=iters_used_0
        ))

        print("Iterations for tile 0:", iters_used_0)
        tile_0.tile.set_learning_rate(lr_save)  # type: ignore

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "program", "O(1)", "very_low", times=2
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "program", "O(n)", "medium",
            vector_length=self.x_size, times=max_iter_zero,
        ))
        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "program", "O(n)", "medium",
            vector_length=self.d_size, times=max_iter_zero,
        ))

        iters_used = []
        # learn tile 0 values for the rest of the tiles
        for weight_tile in self.weight_tiles[1:]:
            lr_save = weight_tile.tile.get_learning_rate()  # type: ignore
            weight_tile.tile.set_learning_rate(learning_rate)  # type: ignore
            target_var = tolerance**2
            variance_threshold = target_var + noise_var * 2 
            alpha = 0.01  # Smoothing factor
            ewmv = variance_threshold * 2  # Initialize above threshold
            counter = 0
            for _ in range(max_iter):
                x_in = torch.randn(self.x_size)
                x_in = x_in / norm_factor
                y = weight_tile.tile.forward(x_in)
                target_values = tile_0.tile.forward(x_in)
                error = y - target_values
                current_var = torch.var(error).item()
                ewmv = (1 - alpha) * ewmv + alpha * current_var
                if ewmv < variance_threshold:
                    break
                weight_tile.update(x_in, error)  # type: ignore
                counter += 1
            iters_used.append(counter)

            weight_tile.tile.set_learning_rate(lr_save)  # type: ignore
        
        total_iter = sum(iters_used)
        max_iters = max(iters_used) if self.num_tiles > 1 else 0
        print("Average iterations:", total_iter / len(iters_used) if len(iters_used) > 0 else 0)


        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "forward",
                    "Meta_op": "program",
                    "Times": iters_used_0,
                    "Pfactor": 1,
                    })
        
        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "update",
                    "Meta_op": "program",
                    "Times": iters_used_0,
                    "Pfactor": 1,
                    })

        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "forward",
                    "Meta_op": "program",
                    "Times": total_iter + max_iters,
                    "Pfactor": total_iter / max_iters if max_iters > 0 else 1,
                    })
        
        self.log.append({
                    "Type": "analog",
                    "Tile": "main",
                    "Op": "update",
                    "Meta_op": "program",
                    "Times": total_iter,
                    "Pfactor": total_iter / max_iters if max_iters > 0 else 1,
                    })

        self.dlog.append(self.create_log_entry(
            "digital_la", "loop_overhead", "program", "O(1)", "low", times=total_iter
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "getter_setter", "program", "O(1)", "very_low", times=2*self.num_tiles
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "program", "O(1)", "very_low", times=4*(self.num_tiles-1)
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "randn", "program", "O(n)", "high", vector_length=self.x_size, times=max_iters
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-element", "program", "O(n)", "medium", vector_length=self.x_size, times=max_iters
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "program", "O(n)", "medium",
            vector_length=self.x_size, times=total_iter
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "pulse-generation", "program", "O(n)", "medium",
            vector_length=self.d_size, times=total_iter,
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "vector-add", "program", "O(n)", "medium", vector_length=self.d_size, times=total_iter
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "variance", "program", "O(n)", "medium", vector_length=self.d_size, times=total_iter
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "arithmetic", "program", "O(1)", "very_low", times=total_iter * 4
        ))

        self.dlog.append(self.create_log_entry(
            "digital_la", "comparison", "program", "O(1)", "very_low", times=total_iter
        ))

    def get_x_size(self) -> int:
        """Returns input size of tile"""
        return self.weight_tiles[0].tile.get_x_size()

    def get_d_size(self) -> int:
        """Returns output size of tile"""
        return self.weight_tiles[0].tile.get_d_size()

    def get_hidden_parameters(self) -> Tensor:
        """Get the hidden parameters of the tile.
        Returns:
        Hidden parameter tensor.
        """
        values_0 = [grad_tile.tile.get_hidden_parameters() for grad_tile in self.grad_tiles]
        
        # Get hidden parameters for each weight tile individually
        values_1 = [w.tile.get_hidden_parameters() for w in self.weight_tiles]
        lst = [
            *values_0,
            *values_1,
            *[grad_tile.tile.get_weights()[None, :, :] for grad_tile in self.grad_tiles],
            *[w.tile.get_weights()[None, :, :] for w in self.weight_tiles],
        ]
        for name, buffer in self.named_buffers():
            if "weight" in name:
                buf = buffer.clone().cpu()
                if self.device_config.transfer_columns:
                    buf = buf.mT
                if len(buf.shape) == 2:  # 2D tensor (single matrix)
                    lst.append(buf[None, :, :])  # Add batch dimension -> (1, rows, cols)
                # else:  # 3D tensor (batch of matrices)
                #     lst.append(buf) # compress to 2D and add batch dimension -> (1, batch_size, rows*cols)

        return torch.concatenate(lst, axis=0)

    def get_hidden_parameter_names(self) -> List[str]:
        """Get the hidden parameters names.
        Each name corresponds to a slice in the Tensor slice of the
        ``get_hidden_parameters`` tensor.
        
        Returns:
        List of names.
        """
        # Hidden parameter names for grad tile
        names_0 = []
        for i, grad_tile in enumerate(self.grad_tiles, 1):
            tile_names = grad_tile.tile.get_hidden_parameter_names()
            names_0.extend([f"{n}_fast_{i}" for n in tile_names])

        
        # Hidden parameter names for weight tiles
        names_1 = []
        for i, w_tile in enumerate(self.weight_tiles, 1):
            tile_names = w_tile.tile.get_hidden_parameter_names()
            names_1.extend([f"{n}_slow_{i}" for n in tile_names])
        
        # Construct final list of names
        lst = (
            names_0 + 
            names_1 + 
            [f"fast_weight_{i}" for i in range(1, len(self.grad_tiles)+1)] +
            [f"slow_weight_{i}" for i in range(1, len(self.weight_tiles)+1)]
        )
        
        # Add additional buffer names
        for name, _ in self.named_buffers():
            # if "weight" in name:
            if "hidden_weight" in name:
                lst.append(name)
        
        return lst

    def set_hidden_parameters(self, params: Tensor) -> None:
        """Set the hidden parameters of the tiles."""
        names = self.get_hidden_parameter_names()
        
        # Get number of base parameters
        n_base_params_0 = len(self.grad_tiles[0].tile.get_hidden_parameter_names())
        n_base_params_1 = len(self.weight_tiles[0].tile.get_hidden_parameter_names())
        
        # Set grad tile hidden parameters
        for i, grad_tile in enumerate(self.grad_tiles):
            start_idx = i * n_base_params_0
            end_idx = start_idx + n_base_params_0
            values_0 = params[start_idx:end_idx, :, :]
            grad_tile.tile.set_hidden_parameters(values_0)
        
        # Set weight tiles hidden parameters
        for i, w_tile in enumerate(self.weight_tiles):
            start_idx = n_base_params_0 + i * n_base_params_1
            end_idx = start_idx + n_base_params_1
            values_1 = params[start_idx:end_idx, :, :]
            w_tile.tile.set_hidden_parameters(values_1)
        
        # Set weights
        for i, grad_tile in enumerate(self.grad_tiles, 1):
            fast_weight_name = f"fast_weight_{i}"
            grad_tile.tile.set_weights(params[names.index(fast_weight_name)])

        # Set weights for each tile
        for i, w_tile in enumerate(self.weight_tiles, 1):
            slow_weight_name = f"slow_weight_{i}"
            w_tile.tile.set_weights(params[names.index(slow_weight_name)])
        
        # Set additional buffers
        # for name, buffer in self.named_buffers():
        #     if name in names:
        #         if len(params[names.index(name), :, :].shape) == 3:
        #             weight = params[names.index(name), :, :, :]
        #         else:
        #             weight = params[names.index(name), :, :]
        #         if self.device_config.transfer_columns:
        #             weight = weight.mT
        #         buffer.data = weight
        
        for name, buffer in self.named_buffers():
            if name in names:
                weight = params[names.index(name), :, :]
                if self.device_config.transfer_columns:
                    weight = weight.T
                buffer.data = weight

    def get_learning_rate(self) -> Optional[float]:
        """Get the learning rate of the tile.

        Returns:
           learning rate if exists.
        """
        return self.learning_rate

    def set_learning_rate(self, learning_rate: Optional[float]) -> None:
        """Set the learning rate of the tile.

        No-op for tiles that do not need a learning rate.

        Args:
           learning rate: learning rate to set
        """
        if learning_rate is None:
            return
        self.learning_rate = learning_rate

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """
        [grad_tile.post_update_step() for grad_tile in self.grad_tiles]
        [weight_tile.post_update_step() for weight_tile in self.weight_tiles]

    def dump_extra(self) -> Optional[Dict]:
        """Dumps any extra states / attributed necessary for
        checkpointing.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.
        """
        return {
            "weight_tiles": [weight_tile.tile.dump_extra() for weight_tile in self.weight_tiles],
            "grad_tiles": [grad_tile.tile.dump_extra() for grad_tile in self.grad_tiles],
        }

    def load_extra(self, extra: Dict, strict: bool = False) -> None:
        """Load any extra states / attributed necessary for
        loading from checkpoint.

        For Tiles based on Modules, this should be normally handled by
        torch automatically.

        Note:
            Expects the exact same RPUConfig / device etc for applying
            the states. Cross-loading of state-dicts is not supported
            for extra states, they will be just ignored.

        Args:
            extra: dictionary of states from `dump_extra`.
            strict: Whether to throw an error if keys are not found.

        Raises:
            RuntimeError: in case keys are wrong
        """

        if "grad_tiles" not in extra or "weight_tiles" not in extra:
            raise RuntimeError("Wrong keys")
        self.weight_tiles.load_extra(extra["weight_tiles"], strict)
        self.grad_tiles.load_extra(extra["grad_tiles"], strict)

    def set_weights_uniform_random(self, bmin: float, bmax: float) -> None:
        """Sets the weights to uniform random numbers.

        Args:
           bmin: min value
           bmax: max value
        """
        [weight_tile.tile.set_weights_uniform_random(bmin, bmax) for weight_tile in self.weight_tiles]

    def get_meta_parameters(self) -> Any:
        """Returns meta parameters."""
        return [weight_tile.tile.get_meta_parameters() for weight_tile in self.weight_tiles]


class TorchTransferTile(TileModule, MultiTileWithPeriphery, SimulatorTileWrapper):
    r"""Transfer tile for in-memory gradient accumulation algorithms.

    This is a (mostly) python re-implemetation of the
    :class:`~aihwkit.simulator.tiles.analog.AnalogTile` with
    :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`` that is using the
    C++ RPUCuda library.

    Here only a subset of the parameters are implemented. However, all
    :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound`` as well as
    :class:`~aihwkit.simulator.configs.compounds.DynamicTransferCompound`` are implemented
    here.

    Thus, TTv2, c-TTv2, and AGAD learning algorithms are implemented here.

    Note:

        This implementation is for instructive use mostly. The C++ implementation has
        large speed advantage if the batch size is large and transfer is done multiple
        times per batch. For the torch implementation, at `transfer_every` needs to be
        larger or same size as the batch size, so that only one transfer is made per
        batch.

    Caution:

        When using ``model.analog_tiles()`` generators, this parent
        tile as well as the children tiles will be looped over, which
        might cause in e.g. getting the same weight twice. This is
        because ``TorchTransferTile`` is registered separately as an
        ``TileModule`` to support the periphery, while internally two
        additional tiles are instantiated.

    Usage::

        rpu_config = build_config('agad', device_config)
        # use the torch implementation tile instead of the default RPUCuda with AnalogTile
        rpu_config.tile_class = TorchTransferTile


    Args:
        out_size: output vector size of the tile, ie. the dimension of
            :math:`\mathbf{y}` in case of :math:`\mathbf{y} =
            W\mathbf{x}` (or equivalently the dimension of the
            :math:`\boldsymbol{\delta}` of the backward pass).
        in_size: input vector size, ie. the dimension of the vector
            :math:`\mathbf{x}` in case of :math:`\mathbf{y} =
            W\mathbf{x}`).

        rpu_config: resistive processing unit configuration. This has to be of type
            :class:`~aihwkit.simulator.configs.configs.UnitCellRPUConfig` with a device
            compound derived from
            :class:`~aihwkit.simulator.configs.compounds.ChoppedTransferCompound``.

        bias: whether to add a bias column to the tile, ie. :math:`W`
            has an extra column to code the biases. This is not supported here.
        in_trans: Whether to assume an transposed input (batch first). Not supported
        out_trans: Whether to assume an transposed output (batch first). Not supported

    Raises:
        ConfigError: if one of the not supported cases is used.

    """

    supports_indexed: bool = False
    supports_ddp: bool = False

    def __init__(
        self,
        in_size: int,
        out_size: int,
        rpu_config: RPUConfigGeneric,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        TileModule.__init__(self)
        SimulatorTileWrapper.__init__(
            self, out_size, in_size, rpu_config, bias, in_trans, out_trans, ignore_analog_state=True
        )
        MultiTileWithPeriphery.__init__(self)

    def _create_simulator_tile(
        self, x_size: int, d_size: int, rpu_config: RPUConfigGeneric
    ) -> BatchedTransferSimulatorTile:
        """Create a simulator tile.

        Args:
            x_size: input size
            d_size: output size
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """

        if not isinstance(rpu_config, UnitCellRPUConfig):
            raise ConfigError("Expect an UnitCellRPUConfig.")

        return BatchedTransferSimulatorTile(x_size, d_size, rpu_config, dtype=self.get_data_type())

    def forward(
        self, x_input: Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> Tensor:
        """Torch forward function that calls the analog forward"""
        # pylint: disable=arguments-differ

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out

    @no_grad()
    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Note:
            This function is called by the analog optimizer.

        Caution:
            If no analog optimizer is used, the post update steps will
            not be performed.
        """
        self.tile.post_update_step()

    def replace_with(self, rpu_config: RPUConfigBase) -> None:
        """Replacing the current `RPUConfig` is not supported.

        Args:
            rpu_config: New `RPUConfig` to check against

        Raises:
            TileModuleError: always
        """
        raise NotImplementedError
