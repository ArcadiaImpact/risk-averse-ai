"""In-process-of-the-worker activation steering for the vLLM (V1) backend.

Why this is not a simple forward hook from the caller
-----------------------------------------------------
A residual-stream steering vector is not a weight delta, so it cannot be "merged"
into the model the way a LoRA adapter can. It must be applied during the forward
pass. But vLLM's V1 engine runs the model in a **separate worker process**, so a
forward hook registered in the caller's process never fires. The supported way to
touch the worker's model is `LLM.apply_model(fn)` / `LLM.collective_rpc(fn)`, which
run `fn(model)` inside the worker. Passing a Python function requires
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` (the default serializer rejects arbitrary
callables); that flag just permits pickling our own trusted code.

So we use `apply_model` to (a) register a forward hook on the steering layer inside
the worker, storing mutable state on the model, and (b) update the steering alpha
between sweep passes. The engine must be built with `enforce_eager=True` so the
worker runs the model eagerly and the hook actually fires (CUDA graphs would bypass
it).

Apply mode is all-positions (Contrastive Activation Addition / CAA): the vector is
added to every token's residual at the chosen layer. This is the standard CAA
semantics and the only apply mode well defined under vLLM's flattened/paged batches.

Equivalence: in vLLM a Qwen3 decoder layer returns `(hidden_states, residual)`; the
residual stream after block L is `hidden_states + residual`, so adding `alpha*v` to
element [0] increases the residual stream by exactly `alpha*v` — the same
intervention the transformers `ResidualSteeringHook` performs. Always validate with
the vLLM-vs-transformers self-check before trusting numbers.
"""

from __future__ import annotations

import os
from functools import partial

import torch


def _enable_insecure_serialization():
    # Permit apply_model/collective_rpc to ship our own functions to the worker.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _decoder_layers(model):
    # Standard (`.model.layers`), GPT-style (`.transformer.h`), and multimodal
    # wrappers like Gemma-3 whose decoder blocks live under a nested text submodule
    # (`.language_model.model.layers` in vLLM, `.model.language_model.layers` in transformers).
    candidates = (
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.language_model.model.layers,   # Gemma-3 multimodal (vLLM)
        lambda m: m.model.language_model.layers,   # Gemma-3 multimodal (transformers)
        lambda m: m.language_model.layers,
    )
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError(
        f"Unsupported model architecture for steering: {type(model).__name__} "
        "(expected `.model.layers`, `.transformer.h`, or a nested `.language_model` text module)."
    )


def _worker_register(model, direction_list, layer_index, apply_mode):
    """Runs inside the vLLM worker: register an all-positions steering hook."""
    import torch as _torch  # worker-local import

    layers = _decoder_layers(model)
    n_layers = len(layers)
    if not (0 <= layer_index < n_layers):
        raise ValueError(f"steering layer {layer_index} out of range (model has {n_layers}).")

    prev = getattr(model, "_steer_state", None)
    if prev is not None and prev.get("handle") is not None:
        prev["handle"].remove()

    state = {"alpha": 0.0, "dir": None, "dir_list": direction_list, "apply_mode": apply_mode}

    def hook(_module, _inp, out):
        a = state["alpha"]
        if a == 0.0:
            return out
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        d = state["dir"]
        if d is None or d.device != h.device or d.dtype != h.dtype:
            d = _torch.tensor(state["dir_list"], dtype=h.dtype, device=h.device)
            state["dir"] = d
        h2 = h + a * d  # broadcasts (hidden,) over [..., hidden]; all positions (CAA)
        if is_tuple:
            return (h2,) + tuple(out[1:])
        return h2

    state["handle"] = layers[layer_index].register_forward_hook(hook)
    model._steer_state = state
    return (n_layers, type(model).__name__)


def _worker_set_alpha(model, alpha):
    st = getattr(model, "_steer_state", None)
    if st is not None:
        st["alpha"] = float(alpha)
        return st["alpha"]
    return None


def _worker_n_layers(model):
    return len(_decoder_layers(model))


def _first(result):
    """apply_model returns a per-worker list; take the first (tp=1)."""
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def get_vllm_n_layers(engine) -> int:
    _enable_insecure_serialization()
    return int(_first(engine.apply_model(_worker_n_layers)))


class VLLMSteeringController:
    """Caller-side handle to the worker-resident steering hook.

    Keep one controller across an alpha sweep and call ``set_alpha(a)`` before each
    pass. ``alpha == 0`` is a true no-op (clean baseline).
    """

    def __init__(self, engine, layer_index, apply_mode="all_positions"):
        self.engine = engine
        self.layer_index = layer_index
        self.apply_mode = apply_mode
        self.alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)
        _enable_insecure_serialization()
        self.engine.apply_model(partial(_worker_set_alpha, alpha=self.alpha))

    def remove(self) -> None:
        _enable_insecure_serialization()
        self.engine.apply_model(partial(_worker_set_alpha, alpha=0.0))


def attach_vllm_steering(engine, direction, layer_index, alpha=0.0, apply_mode="all_positions"):
    """Register an all-positions CAA steering hook on `layer_index` inside the worker."""
    if apply_mode != "all_positions":
        raise ValueError(f"vLLM steering supports only apply_mode='all_positions' (CAA), got {apply_mode!r}.")
    _enable_insecure_serialization()
    direction_list = direction.detach().to(torch.float32).cpu().tolist()
    n_layers, model_name = _first(
        engine.apply_model(
            partial(_worker_register, direction_list=direction_list, layer_index=layer_index, apply_mode=apply_mode)
        )
    )
    controller = VLLMSteeringController(engine, layer_index, apply_mode)
    controller.n_layers = n_layers
    controller.model_name = model_name
    controller.set_alpha(alpha)
    return controller
