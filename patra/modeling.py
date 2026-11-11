"""Loading PATRA checkpoints and building the LLM input embeddings.

A single model replica is split across GPUs layer by layer (``accelerate`` device map). The TS encoder, the
alignment module and the token embeddings are kept on one device so the time-series path never crosses GPUs.

``merge_ts_embeddings`` is the batch-size-1 part of ``Qwen2TSForCausalLM.forward``: text embeddings -> TS encoder ->
pattern-aware alignment -> splice the aligned TS tokens into the sequence. Running it per sample and batching only
the LLM is what makes batched evaluation/training correct: the released ``forward`` assumes no left padding and runs
the alignment's moving average over the patches of the whole batch.
"""
import torch
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

TS_MODULES = ('ts_encoder', 'align')


def unwrap(model):
    """The underlying ``Qwen2TSForCausalLM`` of a (possibly PEFT-wrapped) model."""
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def embedding_device(model):
    return model.get_input_embeddings().weight.device


def parse_max_memory(spec):
    """``'0:20GiB,1:8GiB'`` -> ``{0: '20GiB', 1: '8GiB', 'cpu': '0GiB'}`` (never offload to CPU)."""
    mm = {int(k): v for k, v in (kv.split(':') for kv in spec.split(','))}
    mm['cpu'] = '0GiB'
    return mm


def build_device_map(model_path, max_memory, dtype, ts_device=None):
    """Layer-wise device map; embeddings + TS modules go to ``ts_device`` (default: wherever the embeddings land)."""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
    dm = infer_auto_device_map(empty, max_memory=max_memory, no_split_module_classes=['Qwen2DecoderLayer'], dtype=dtype)
    first = dm.get('model.embed_tokens', dm.get('model', 0)) if ts_device is None else ts_device
    for k in TS_MODULES + ('model.embed_tokens',):
        dm[k] = first
    return dm


def reinit_ts_modules(model, seed):
    """Fresh TS encoder + alignment, as ``from_pretrained`` would initialise missing weights
    (normal(0, initializer_range) for matrices, zeros for vectors). Used because PATRA-init is not released."""
    g = torch.Generator().manual_seed(seed)
    std = model.config.initializer_range
    params = list(model.ts_encoder.named_parameters()) + list(model.align.named_parameters())
    with torch.no_grad():
        for _, p in params:
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).mul_(std).to(p.dtype))
            else:
                p.zero_()


def load_model(model_path, max_memory=None, device=None, dtype=torch.bfloat16, ts_device=None,
               reinit_ts=False, seed=0):
    """Load a PATRA-architecture checkpoint (or a plain causal LM) and its tokenizer.

    Exactly one of ``max_memory`` (split over GPUs, e.g. ``'0:20GiB'``) or ``device`` (e.g. ``'cuda:0'``) should be
    given; with neither, ``device_map='auto'`` is used.
    """
    if device is not None:
        device_map = device
    elif max_memory is not None:
        device_map = build_device_map(model_path, parse_max_memory(max_memory), dtype, ts_device)
    else:
        device_map = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device_map,
                                                 torch_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    if reinit_ts:
        reinit_ts_modules(model, seed)
    return model, tok


def load_processor(model_path, tokenizer):
    """The checkpoint's own ``Qwen2TSProcessor`` (official tokenisation + ``sp_encoding``)."""
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True, tokenizer=tokenizer)


def merge_ts_embeddings(model, input_ids, ts=None):
    """Input embeddings ``[L, D]`` for one prompt, with the aligned TS tokens spliced in.

    input_ids: ``[1, L]`` token ids containing one ``<ts>`` start token per series; ts: ``[n_series, len, 1]`` or None.
    Differentiable with respect to the TS encoder, the alignment module and any LoRA weights.
    """
    base = unwrap(model)
    dev = embedding_device(model)
    input_ids = input_ids.to(dev)
    embeds = model.get_input_embeddings()(input_ids)
    if ts is None or ts.shape[0] == 0:
        return embeds[0]
    feats, patch_counts = base.ts_encoder(ts.to(dev, dtype=base.ts_encoder.mlp[0].weight.dtype))
    embeds = embeds.to(feats.dtype)
    aligned, aligned_counts = base.align(input_ids, embeds, feats, patch_counts)
    aligned = aligned.reshape(-1, aligned.shape[-1])
    attn = torch.ones_like(input_ids)
    embeds, attn, _, _ = base._merge_input_ids_with_time_series_features(aligned, embeds, input_ids, attn, None,
                                                                          aligned_counts)
    assert attn.all(), 'unexpected padding in single-sample merge'
    return embeds[0]


def prompt_embedding(model, tok, prompt, ts=None):
    """Tokenise ``prompt`` (no special tokens added) and build its input embeddings. Returns ``(embeds, ids)``."""
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
    if ts is not None:
        n_ph = prompt.count('<ts><ts/>')
        assert n_ph == ts.shape[0], f'{n_ph} placeholders vs {ts.shape[0]} series'
    return merge_ts_embeddings(model, ids, ts), ids[0]


def processor_embedding(model, processor, prompt, timeseries):
    """Same as ``prompt_embedding`` but tokenised/normalised by the checkpoint's processor (the official eval path)."""
    inputs = processor(text=[prompt], timeseries=timeseries, padding=True, return_tensors='pt')
    assert inputs['attention_mask'].all()
    ts = inputs.get('timeseries')
    return merge_ts_embeddings(model, inputs['input_ids'], ts), inputs['input_ids'][0]
