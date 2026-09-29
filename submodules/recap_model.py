"""Independent implementation of the RLinf-style patch-token RECAP value model.

SigLIP2 patch tokens and task tokens form a Gemma3 prefix. A small Gemma
expert reads its KV cache with one learned CLS query. No RLinf import is used.
"""
from contextlib import nullcontext
import string

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Gemma3ForCausalLM, SiglipVisionModel
from transformers.cache_utils import DynamicCache
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaModel

from submodules.contracts import resolve_path


ARCHITECTURE = 'recap_patch_gemma_expert'
EXPERT_VARIANTS = {
    'gemma_1m': dict(hidden_size=128, intermediate_size=448, num_hidden_layers=4,
                     num_attention_heads=1, num_key_value_heads=1, head_dim=256),
    'gemma_50m': dict(hidden_size=384, intermediate_size=1536, num_hidden_layers=18,
                      num_attention_heads=8, num_key_value_heads=1, head_dim=256),
    'gemma_100m': dict(hidden_size=512, intermediate_size=2048, num_hidden_layers=18,
                       num_attention_heads=8, num_key_value_heads=1, head_dim=256),
    'gemma_150m': dict(hidden_size=640, intermediate_size=2560, num_hidden_layers=18,
                       num_attention_heads=8, num_key_value_heads=1, head_dim=256),
    'gemma_300m': dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=18,
                       num_attention_heads=8, num_key_value_heads=1, head_dim=256),
    'gemma_2b': dict(hidden_size=2048, intermediate_size=16384, num_hidden_layers=18,
                     num_attention_heads=8, num_key_value_heads=1, head_dim=256),
}


def two_hot_loss(logits, targets, atoms):
    """RLinf's linear projection of a scalar target onto adjacent value atoms."""
    if logits.ndim != 2 or atoms.ndim != 1 or logits.shape[1] != len(atoms) or len(atoms) < 2:
        raise ValueError('Logits and value atoms must have shapes [batch, bins] and [bins]')
    targets = targets.float().reshape(-1)
    if len(targets) != logits.shape[0] or not torch.isfinite(targets).all():
        raise ValueError('Targets must be finite and match the logits batch')
    targets = targets.clamp(atoms[0], atoms[-1])
    position = (targets - atoms[0]) / (atoms[1] - atoms[0])
    lower = position.floor().long().clamp(0, len(atoms) - 1)
    upper = position.ceil().long().clamp(0, len(atoms) - 1)
    upper_weight = position - lower.float()
    lower_weight = 1.0 - upper_weight
    log_probs = F.log_softmax(logits.float(), dim=-1)
    rows = torch.arange(len(targets), device=logits.device)
    return -(lower_weight * log_probs[rows, lower] +
             upper_weight * log_probs[rows, upper]).mean()


def tokenize_task_prompts(tokenizer, prompts, max_length, device, cache=None):
    """RLinf-style task prefix with right padding and an explicit validity mask."""
    if not prompts or max_length < 1:
        raise ValueError('Nonempty prompts and a positive max token length are required')
    cache = {} if cache is None else cache
    token_rows, mask_rows = [], []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Every image needs a nonempty task prompt')
        cleaned = prompt.lower().strip().replace('_', ' ').replace('\n', ' ')
        if cleaned[-1] in string.punctuation and cleaned[-1] not in "\"'":
            cleaned = cleaned[:-1]
        prefix = f'Task: {cleaned}.'
        if prefix not in cache:
            cache[prefix] = tokenizer.encode(prefix, add_special_tokens=True)
        ids = cache[prefix][:max_length]
        valid = len(ids)
        token_rows.append(ids + [tokenizer.pad_token_id] * (max_length - valid))
        mask_rows.append([True] * valid + [False] * (max_length - valid))
    return (torch.tensor(token_rows, dtype=torch.long, device=device),
            torch.tensor(mask_rows, dtype=torch.bool, device=device))


class RecapValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if int(config['num_bins']) < 2:
            raise ValueError('num_bins must be at least 2')
        self.config = dict(config)
        self.cameras = tuple(config['cameras'])
        self.freeze_vision_encoder = bool(config.get('freeze_vision_encoder', False))
        self.freeze_vlm = bool(config.get('freeze_vlm', False))
        self.freeze_value_expert = bool(config.get('freeze_value_expert', False))
        dtype = torch.bfloat16 if config['precision'] == 'bf16' else torch.float32
        siglip_path = resolve_path(config['siglip_path'])
        gemma_path = resolve_path(config['gemma3_path'])
        self.vision_tower = SiglipVisionModel.from_pretrained(
            siglip_path, local_files_only=True, dtype=dtype, attn_implementation='sdpa')
        self.gemma3 = Gemma3ForCausalLM.from_pretrained(
            gemma_path, local_files_only=True, dtype=dtype, attn_implementation='sdpa')
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolve_path(config.get('tokenizer_path', config['gemma3_path'])), local_files_only=True)
        self._prompt_token_cache = {}
        vision_width = self.vision_tower.config.hidden_size
        vlm_width = self.gemma3.config.hidden_size
        variant = config.get('critic_expert_variant', 'gemma_1m')
        if variant not in EXPERT_VARIANTS:
            raise ValueError(f'Unsupported critic expert: {variant}')
        expert_config = GemmaConfig(vocab_size=257152, hidden_activation='gelu_pytorch_tanh',
                                     **EXPERT_VARIANTS[variant])
        if expert_config.head_dim != self.gemma3.config.head_dim:
            raise ValueError('Expert and Gemma3 KV head dimensions differ')
        self.prefix_depth = expert_config.num_hidden_layers
        if self.prefix_depth > len(self.gemma3.model.layers):
            raise ValueError('Expert cannot read more KV layers than the Gemma3 backbone provides')
        # Expert layer i reads prefix KV layer i, as in RLinf. A shallow debug
        # expert cannot use later prefix layers, so skip their forward computation.
        self.gemma3.config.num_hidden_layers = self.prefix_depth
        for layer in self.gemma3.model.layers[self.prefix_depth:]:
            layer.requires_grad_(False)
        self.expert = GemmaModel(expert_config).to(dtype=dtype)
        self.expert.embed_tokens = None
        self.image_projection = nn.Linear(vision_width, vlm_width).to(dtype=dtype)
        nn.init.normal_(self.image_projection.weight, std=0.02)
        nn.init.zeros_(self.image_projection.bias)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, expert_config.hidden_size, dtype=dtype) * 0.02)
        self.value_projection = nn.Linear(expert_config.hidden_size, config['num_bins']).to(dtype=dtype)
        self.register_buffer('atoms', torch.linspace(-1., 0., config['num_bins']), persistent=False)
        if dtype == torch.bfloat16 and config.get('bf16_keep_fp32_layers', False):
            keep_fp32 = ('embeddings.patch_embedding.weight', 'embeddings.patch_embedding.bias',
                         'embeddings.position_embedding.weight', 'input_layernorm',
                         'post_attention_layernorm', 'model.norm')
            for name, parameter in self.named_parameters():
                if any(part in name for part in keep_fp32):
                    parameter.data = parameter.data.float()
        if self.freeze_vision_encoder:
            self.vision_tower.requires_grad_(False)
        if self.freeze_vlm:
            self.gemma3.requires_grad_(False)
        if self.freeze_value_expert:
            self.expert.requires_grad_(False)
        self.gradient_checkpointing_enabled = False
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_tower.eval()
        if self.freeze_vlm:
            self.gemma3.eval()
        if self.freeze_value_expert:
            self.expert.eval()
        return self

    def gradient_checkpointing_enable(self):
        # HF text-layer checkpointing clears past_key_values. Checkpoint pure
        # prefix/expert stages instead, passing tensor KV tuples between them.
        self.gradient_checkpointing_enabled = True
        if not self.freeze_vision_encoder:
            self.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.vision_tower.gradient_checkpointing_disable()

    def _run_stage(self, function, *args):
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def _prefix_kv(self, prefix, attention, positions):
        out = self.gemma3.model(inputs_embeds=prefix, attention_mask=attention,
                               position_ids=positions, past_key_values=DynamicCache(), use_cache=True)
        return out.past_key_values.to_legacy_cache()

    def _expert_hidden(self, cls, attention, positions, kv):
        # Expert attention appends its own K/V: never mutate the prefix cache or
        # a cache captured by a checkpoint closure across recomputations.
        cache = DynamicCache.from_legacy_cache(kv)
        out = self.expert(inputs_embeds=cls, attention_mask=attention,
                          position_ids=positions, past_key_values=cache, use_cache=False)
        return out.last_hidden_state[:, -1, :]

    def _encode_image(self, image):
        with torch.no_grad() if self.freeze_vision_encoder else nullcontext():
            patches = self.vision_tower(pixel_values=image).last_hidden_state
        return self.image_projection(patches.to(self.image_projection.weight.dtype))

    def forward(self, images, prompts, return_distribution=False, image_masks=None):
        device = next(iter(images.values())).device
        enabled = device.type == 'cuda' and self.config['precision'] == 'bf16'
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            return self._forward(images, prompts, return_distribution, image_masks)

    def _forward(self, images, prompts, return_distribution=False, image_masks=None):
        if not isinstance(prompts, (list, tuple)) or not prompts:
            raise ValueError('A prompt string is required for every image')
        keys = [f'observation.images.{cam}' for cam in self.cameras]
        if set(images) != set(keys):
            raise ValueError(f'Camera mismatch: expected {keys}, got {list(images)}')
        batch_size = len(prompts)
        pieces = [self._encode_image(images[key]) for key in keys]
        if any(piece.shape[0] != batch_size for piece in pieces):
            raise ValueError('Image and prompt batch sizes differ')
        image_masks = {} if image_masks is None else image_masks
        if set(image_masks) - set(keys):
            raise ValueError(f'Unexpected camera masks: {set(image_masks) - set(keys)}')
        image_padding = []
        for key, piece in zip(keys, pieces):
            valid = image_masks.get(key)
            if valid is None:
                valid = torch.ones(batch_size, dtype=torch.bool, device=piece.device)
            else:
                valid = torch.as_tensor(valid, device=piece.device)
                if valid.shape != (batch_size,) or valid.dtype != torch.bool:
                    raise ValueError(f'Camera mask for {key} must be boolean [batch]')
            image_padding.append(valid[:, None].expand(batch_size, piece.shape[1]))
        text_ids, text_mask = tokenize_task_prompts(
            self.tokenizer, prompts, int(self.config.get('max_token_len', 200)),
            pieces[0].device, self._prompt_token_cache)
        pieces.append(self.gemma3.model.embed_tokens(text_ids))
        prefix = torch.cat(pieces, dim=1).to(self.gemma3.model.embed_tokens.weight.dtype)
        padding = torch.cat([*image_padding, text_mask], dim=1)
        prefix_mask = padding[:, None, :] & padding[:, :, None]
        mask_dtype = self.gemma3.model.layers[0].self_attn.q_proj.weight.dtype
        prefix_attention = torch.where(prefix_mask[:, None], 0., torch.finfo(mask_dtype).min).to(mask_dtype)
        kv = self._run_stage(self._prefix_kv, prefix, prefix_attention, padding.long().cumsum(1)-1)
        cls = self.cls_embedding.expand(batch_size, -1, -1)
        suffix_padding = torch.cat([padding, torch.ones(batch_size, 1, dtype=torch.bool, device=prefix.device)], 1)
        suffix_attention = torch.where(suffix_padding[:, None, None, :],
                                       0., torch.finfo(mask_dtype).min).to(mask_dtype)
        hidden = self._run_stage(self._expert_hidden, cls, suffix_attention,
                                 padding.long().sum(1, keepdim=True), kv)
        logits = self.value_projection(hidden).float()
        probabilities = logits.softmax(-1)
        value = (probabilities * self.atoms).sum(-1, keepdim=True)
        if return_distribution:
            return value, logits
        return value
