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

from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults
from nemo.collections.nlp.modules.common.megatron.transformer import AutocastTransformerLayer

try:
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

    ModuleSpec = ApexGuardDefaults


class TETransformerLayerAutocast(AutocastTransformerLayer):
    def __init__(self, config, layer_number=1, hidden_dropout=None):
        self.config = config

        super().__init__(
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

# Use this spec to use the full Transformer layer from Transformer Engine
def get_gpt_full_te_layer_autocast_spec() -> TransformerBlockSubmodules:
    if not HAVE_MEGATRON_CORE:
        raise ImportError(
            "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )

    return TransformerBlockSubmodules(
        layer_specs=ModuleSpec(module=TETransformerLayerAutocast)
    )
