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

try:
    from megatron.core.transformer.spec_utils import ModuleSpec

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

    ModuleSpec = ApexGuardDefaults

from transformer_engine.pytorch.module import TransformerLayer

# Use this spec to use the full Transformer layer from Transformer Engine
def get_gpt_full_te_layer_spec() -> ModuleSpec:
    if not HAVE_MEGATRON_CORE:
        raise ImportError(
            "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
        )
    return ModuleSpec(module=TransformerLayer)
