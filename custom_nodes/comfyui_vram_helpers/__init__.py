"""Tiny custom node pack: targeted VRAM eviction for multi-stage workflows.

Provides two nodes:

- VRAMUnloadModel — takes a LATENT passthrough and a MODEL to evict,
  drops the model's weights entirely (does NOT move them to CPU RAM), and
  removes it from the loaded-models tracker. Returns the latent unchanged.

- VRAMUnloadClip — takes two CONDITIONING passthroughs (positive + negative)
  and a CLIP to evict. Same drop-to-meta mechanism applied to the CLIP's
  inner `cond_stage_model` + `patcher`. Returns both conditionings unchanged.
  Used to free the text encoder before the UNET loads for sampling (e.g.
  Qwen-Image-Edit: drop the VL encoder after text encode so the larger
  image-edit UNET fits on a 24 GB card).

Why not offload to CPU: in a two-stage MoE workflow (e.g. WAN 2.2 high-noise
followed by low-noise), once stage 1 is done the high-noise expert is never
needed again. Moving ~9.5 GB to system RAM wastes capacity that the next
GGUF dequantization (model_low load) may need. Discarding is cheaper and
reversible by never re-running this prompt — a fresh run reloads from disk
anyway.

How "drop" works: we call `inner.to('meta')` which replaces every parameter
and buffer with a meta tensor (shape/dtype metadata only, zero storage).
The underlying CUDA allocations lose their last reference and PyTorch's
cache allocator reclaims them on the next `torch.cuda.empty_cache()`.

The model becomes unusable after this — any forward pass raises — which
is fine because we never touch it again in the current prompt.

Re-run cache invalidation
-------------------------
After meta-tensoring, we also surgically pop the upstream model-chain
entries from the running PromptExecutor's output cache. Without this,
ComfyUI would reuse the now-meta-tensored ModelPatcher on a subsequent
enqueue (e.g. when only the sampler seed changed) and stage 1's
KSampler would crash with:
    RuntimeError: Tensor.item() cannot be called on meta tensors

We find the executor's caches by walking the Python call stack for the
`caches` local in `execution.execute()` — ComfyUI does not expose the
executor via a module-level singleton, so stack introspection is the
least invasive hook. Then for every ancestor of our `model` input in the
prompt graph, we set the cache entry to None, which makes the next
`caches.outputs.get(node_id)` return None → cache miss → node re-executes.
This leaves every other cache entry hot, so downstream experts and text
encoders are not re-run unnecessarily.

Wiring example (WAN 2.2 two-stage):
    stage1_ksampler_high.latent -> VRAMUnloadModel.latent -> stage2_ksampler_low.latent_image
    modelsampling_high.model    -> VRAMUnloadModel.model
"""
import gc
import inspect

import torch

import comfy.model_management as mm


def _find_executor_caches():
    """Walk the call stack for `caches` in execution.execute()'s frame.

    ComfyUI's `execution.execute()` is an async module-level function that
    receives the PromptExecutor's CacheSet as a parameter. When a node's
    FUNCTION runs, this frame is on the stack (via _async_map_node_over_list
    → get_output_data → execute). Returns the CacheSet or None if not found.
    """
    frame = inspect.currentframe()
    try:
        while frame is not None:
            cand = frame.f_locals.get("caches")
            if cand is not None and hasattr(cand, "outputs"):
                outputs = getattr(cand, "outputs", None)
                if outputs is not None and hasattr(outputs, "set_local"):
                    return cand
            frame = frame.f_back
    finally:
        del frame
    return None


def _collect_model_ancestors(prompt, start_node_id):
    """Return the set of node IDs whose outputs transitively feed `start_node_id`.

    `prompt` is the dict ComfyUI passes as the PROMPT hidden input:
        { "<node_id>": {"class_type": str, "inputs": {name: value | [src_id, slot]}} }
    Input values that are 2-element lists (src_id, slot) are graph edges.
    """
    ancestors: set[str] = set()
    stack: list[str] = [str(start_node_id)]
    while stack:
        nid = stack.pop()
        node = prompt.get(nid) or prompt.get(int(nid) if nid.isdigit() else nid)
        if not isinstance(node, dict):
            continue
        for _, val in (node.get("inputs") or {}).items():
            if isinstance(val, list) and len(val) >= 1:
                src = str(val[0])
                if src not in ancestors:
                    ancestors.add(src)
                    stack.append(src)
    return ancestors


class VRAMUnloadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "model": ("MODEL",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "unload"
    CATEGORY = "vram"

    def unload(self, latent, model, prompt=None, unique_id=None):
        device = mm.get_torch_device()
        before_free = mm.get_free_memory(device)

        inner = getattr(model, "model", None)

        patcher_cls = type(model).__name__
        inner_cls = type(inner).__name__ if inner is not None else "None"
        extra_bits: list[str] = []
        model_config = getattr(inner, "model_config", None)
        if model_config is not None:
            mc_name = getattr(model_config, "__class__", type(model_config)).__name__
            extra_bits.append(f"config={mc_name}")
        model_type = getattr(inner, "model_type", None)
        if model_type is not None:
            extra_bits.append(f"type={model_type}")
        param_bytes = 0
        if inner is not None:
            try:
                for p in inner.parameters():
                    if p.data is not None and p.data.numel() > 0:
                        param_bytes += p.data.numel() * p.data.element_size()
            except Exception:
                pass
        if param_bytes:
            extra_bits.append(f"weights={param_bytes/1e9:.2f} GB")
        ident = f"{patcher_cls}<{inner_cls}>" + (" [" + ", ".join(extra_bits) + "]" if extra_bits else "")

        print(f"[VRAMUnloadModel] dropping: {ident}")

        for attr in ("backup", "object_patches", "object_patches_backup"):
            d = getattr(model, attr, None)
            if isinstance(d, dict):
                try:
                    d.clear()
                except Exception:
                    pass

        _drop_to_meta(inner, "VRAMUnloadModel")
        _remove_from_loaded_models(model, inner)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        mm.soft_empty_cache(True)

        after_free = mm.get_free_memory(device)
        freed_gb = (after_free - before_free) / 1e9
        print(
            f"[VRAMUnloadModel] dropped: {ident} | "
            f"VRAM freed: {freed_gb:+.2f} GB "
            f"(free {before_free/1e9:.2f} -> {after_free/1e9:.2f} GB)"
        )

        _invalidate_input_chain_cache("VRAMUnloadModel", prompt, unique_id, "model")

        return (latent,)


def _drop_to_meta(inner, label: str) -> None:
    """Drop every parameter/buffer of `inner` to a meta tensor (zero storage)."""
    if inner is None:
        return
    try:
        inner.to("meta")
    except Exception as e:
        print(f"[{label}] inner.to('meta') failed: {e}; replacing .data manually")
        for p in list(inner.parameters()):
            try:
                p.data = torch.empty(0, dtype=p.dtype)
            except Exception:
                pass
        for b in list(inner.buffers(recurse=True)):
            try:
                b.data = torch.empty(0)
            except Exception:
                pass


def _remove_from_loaded_models(*patchers_and_inners) -> None:
    """Remove any tracked LoadedModel whose patcher/inner matches one of the refs."""
    refs = {id(obj) for obj in patchers_and_inners if obj is not None}
    if not refs:
        return
    for lm in list(mm.current_loaded_models):
        lm_patcher = getattr(lm, "model", None)
        lm_inner = getattr(lm_patcher, "model", None) if lm_patcher is not None else None
        if id(lm_patcher) in refs or (lm_inner is not None and id(lm_inner) in refs):
            try:
                mm.current_loaded_models.remove(lm)
            except ValueError:
                pass


def _invalidate_input_chain_cache(label: str, prompt, self_node_id, input_name: str):
    """Pop every ancestor of `inputs[input_name]` from the executor output cache.

    On the NEXT enqueue, those nodes re-execute (fresh loader → fresh weights),
    while every cache entry NOT in that input's ancestry stays hot.
    """
    if not isinstance(prompt, dict) or self_node_id is None:
        print(f"[{label}] cache invalidation skipped (missing prompt/unique_id)")
        return

    caches = _find_executor_caches()
    if caches is None:
        print(f"[{label}] cache invalidation skipped (executor caches not on stack)")
        return

    self_node = prompt.get(str(self_node_id)) or prompt.get(self_node_id)
    if not isinstance(self_node, dict):
        return
    link = (self_node.get("inputs") or {}).get(input_name)
    if not (isinstance(link, list) and len(link) >= 1):
        return

    root_src = str(link[0])
    ancestors = _collect_model_ancestors(prompt, root_src)
    ancestors.add(root_src)

    outputs_cache = caches.outputs
    invalidated: list[str] = []
    for nid in ancestors:
        try:
            outputs_cache.set_local(nid, None)
            invalidated.append(nid)
        except AssertionError:
            pass
        except Exception as e:
            print(f"[{label}] failed to invalidate node {nid}: {e}")

    if invalidated:
        ordered = sorted(invalidated, key=lambda x: int(x) if x.isdigit() else x)
        print(f"[{label}] invalidated output cache for nodes: {', '.join(ordered)}")


class VRAMUnloadClip:
    """Drop a CLIP's weights entirely after text encoding completes.

    Takes both conditionings as passthroughs so ComfyUI's topological
    executor runs this node AFTER both TextEncode nodes finish (their
    outputs are required inputs here), and BEFORE any downstream node
    that consumes the conditionings (ControlNetApply, KSampler, …).

    The drop targets both the CLIP's inner `cond_stage_model` (the actual
    text encoder weights, incl. GGUF dequantized tensors) and the
    `patcher` ModelPatcher wrapping it, and removes both from ComfyUI's
    `current_loaded_models` tracker. On the next enqueue, the CLIP loader
    chain re-executes (fresh weights) via the standard cache invalidation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "clip": ("CLIP",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "unload"
    CATEGORY = "vram"

    def unload(self, positive, negative, clip, prompt=None, unique_id=None):
        device = mm.get_torch_device()
        before_free = mm.get_free_memory(device)

        inner = getattr(clip, "cond_stage_model", None)
        patcher = getattr(clip, "patcher", None)

        inner_cls = type(inner).__name__ if inner is not None else "None"
        patcher_cls = type(patcher).__name__ if patcher is not None else "None"
        param_bytes = 0
        if inner is not None:
            try:
                for p in inner.parameters():
                    if p.data is not None and p.data.numel() > 0:
                        param_bytes += p.data.numel() * p.data.element_size()
            except Exception:
                pass
        ident = f"CLIP<{inner_cls}> via {patcher_cls}"
        if param_bytes:
            ident += f" [weights={param_bytes/1e9:.2f} GB]"

        print(f"[VRAMUnloadClip] dropping: {ident}")

        for target in (patcher, clip):
            for attr in ("backup", "object_patches", "object_patches_backup"):
                d = getattr(target, attr, None)
                if isinstance(d, dict):
                    try:
                        d.clear()
                    except Exception:
                        pass

        _drop_to_meta(inner, "VRAMUnloadClip")
        _remove_from_loaded_models(patcher, inner)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        mm.soft_empty_cache(True)

        after_free = mm.get_free_memory(device)
        freed_gb = (after_free - before_free) / 1e9
        print(
            f"[VRAMUnloadClip] dropped: {ident} | "
            f"VRAM freed: {freed_gb:+.2f} GB "
            f"(free {before_free/1e9:.2f} -> {after_free/1e9:.2f} GB)"
        )

        _invalidate_input_chain_cache("VRAMUnloadClip", prompt, unique_id, "clip")

        return (positive, negative)


NODE_CLASS_MAPPINGS = {
    "VRAMUnloadModel": VRAMUnloadModel,
    "VRAMUnloadClip": VRAMUnloadClip,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMUnloadModel": "VRAM Unload Model (drop weights)",
    "VRAMUnloadClip": "VRAM Unload CLIP (drop weights)",
}
