# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults, init_method_normal, scaled_init_method_normal
from nemo.collections.nlp.modules.common.megatron.transformer import AutocastTransformerLayer

try:
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

    ModuleSpec = ApexGuardDefaults


class TETransformerLayerAutocast(AutocastTransformerLayer):
    def __init__(self, config, layer_number=1, hidden_dropout=None):
        self.config = config

        # TODO: Expose knobs instead of hardcoding
        init_method_std = 0.006
        init_method = init_method_normal(init_method_std)
        scaled_init_method = init_method_normal(init_method_std) # Assumes use_scaled_init_method = False

        '''
        # init from nemo/collections/nlp/modules/common/megatron/transformer.py#L1057
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        layernorm_epsilon=layernorm_epsilon,
        num_attention_heads=num_attention_heads,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        hidden_dropout=hidden_dropout,
        attention_dropout=attention_dropout,
        layer_number=layer_number + layer_number_offset,
        kv_channels=kv_channels,
        self_attn_mask_type=self_attn_mask_type.name,
        tp_size=parallel_state.get_tensor_model_parallel_world_size(),
        params_dtype=config.params_dtype,
        get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
        fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
        seq_length=None,  # used for jit warmup
        micro_batch_size=None,  # used for jit warmup
        sequence_parallel=config.sequence_parallel,
        apply_residual_connection_post_layernorm=False,
        autocast_dtype=precision,
        use_emha=use_emha,
        ub_tp_comm_overlap=ub_tp_comm_overlap,
        zero_centered_gamma=normalization == 'layernorm1p',
        device='cpu' if config.use_cpu_initialization else 'cuda',
        '''
        # Currently hardcoded for config_DGXH100_16x8x32x4x8_mbs1.sh
        # TODO: Expose knobs through NeMo instead of hardcoding
        super().__init__(
            hidden_size=12288,
            ffn_hidden_size=49152,
            layernorm_epsilon=1e-05,
            num_attention_heads=96,
            init_method=init_method,
            output_layer_init_method=scaled_init_method,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            #layer_number=9,
            kv_channels=128,
            self_attn_mask_type='causal',
            tp_size=parallel_state.get_tensor_model_parallel_world_size(),
            params_dtype=torch.bfloat16,
            get_rng_state_tracker=tensor_parallel.random.get_cuda_rng_tracker,
            fuse_wgrad_accumulation=True,
            seq_length=None,  # used for jit warmup
            micro_batch_size=None,  # used for jit warmup
            sequence_parallel=True,
            apply_residual_connection_post_layernorm=False,
            autocast_dtype=16,
            use_emha=False,
            ub_tp_comm_overlap=True,
            zero_centered_gamma=True,
            device='cuda',
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
    ):
        hidden_states = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            encoder_output=context,
            enc_dec_attn_mask=context_mask,
            inference_params=inference_params,
            #is_first_microbatch,
            #checkpoint_core_attention,
        )
        context = None

        return hidden_states, context

    def _get_layer_offset(self):

        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = ()):
        TE_LAYERS_MAP = {
            "self_attention.layernorm_qkv.layer_norm_weight": "input_layernorm.weight",
            "self_attention.layernorm_qkv.layer_norm_bias": "input_layernorm.bias",
            "layernorm_mlp.layer_norm_weight": "post_attention_layernorm.weight",
            "layernorm_mlp.layer_norm_bias": "post_attention_layernorm.bias",
            "layernorm_mlp.fc1_weight": "mlp.dense_h_to_4h.weight",
            "layernorm_mlp.fc1_bias": "mlp.dense_h_to_4h.bias",
            "layernorm_mlp.fc2_weight": "mlp.dense_4h_to_h.weight",
            "layernorm_mlp.fc2_bias": "mlp.dense_4h_to_h.bias",
            "self_attention.layernorm_qkv.weight": "self_attention.query_key_value.weight",
            "self_attention.layernorm_qkv.bias": "self_attention.query_key_value.bias",
            "self_attention.proj.weight": "self_attention.dense.weight",
            "self_attention.proj.bias": "self_attention.dense.bias",
        }

        TENSOR_PARALLEL_LAYERS_AXIS_MAP = {
            'self_attention.layernorm_qkv.weight': 0,
            'self_attention.layernorm_qkv.bias': 0,
            "self_attention.proj.weight": 1,
            "layernorm_mlp.fc1_weight": 0,
            "layernorm_mlp.fc1_bias": 0,
            "layernorm_mlp.fc2_weight": 1,
        }

        state_dict = self.state_dict(prefix='', keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(state_dict, prefix, TENSOR_PARALLEL_LAYERS_AXIS_MAP, sharded_offsets)

        prefixed_map = {f'{prefix}{k}': f'{prefix}{v}' for k, v in TE_LAYERS_MAP.items()}

        for k, v in sharded_state_dict.items():
            if not v.key.endswith('_extra_state'):
                v.key = prefixed_map[v.key]

        return sharded_state_dict

# Use this spec to use the full Transformer layer from Transformer Engine
def get_gpt_full_te_layer_autocast_spec() -> TransformerBlockSubmodules:
    if not HAVE_MEGATRON_CORE:
        raise ImportError(
            "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )

    return TransformerBlockSubmodules(
        layer_specs=ModuleSpec(module=TETransformerLayerAutocast)
    )