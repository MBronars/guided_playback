"""Exponential moving averages with strict, checkpointable training state."""
from contextlib import contextmanager
import math
import warnings

import torch


def validate_ema_state(model, state_dict, *, metadata=None, decay=0.9999, warmup=False):
    """Check shadow names, shapes and metadata without changing model or EMA."""
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing = sorted(parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - parameters.keys())
    wrong_shape = [n for n in parameters.keys() & state_dict.keys()
                   if not isinstance(state_dict[n], torch.Tensor)
                   or state_dict[n].shape != parameters[n].shape]
    if missing or unexpected or wrong_shape:
        raise RuntimeError(f"EMA state mismatch: missing={missing}, unexpected={unexpected}, shape={wrong_shape}")
    updates = 0
    if metadata is not None:
        updates, decay, warmup = metadata["updates"], metadata["decay"], metadata["warmup"]
        if (metadata.get("version", 1) != 1 or type(updates) is not int or updates < 0
                or not math.isfinite(decay) or not 0 <= decay <= 1 or type(warmup) is not bool):
            raise RuntimeError(f"Invalid EMA metadata: {metadata}")
    elif warmup:
        updates = 10_000_000
    return parameters, updates, decay, warmup


class EMA:
    """Track trainable parameters; keep inference weights a plain name→tensor map.

    Training checkpoints store metadata separately so historical inference loaders
    can still consume the shadow mapping. Initialize after device placement.
    """

    def __init__(self, model, decay=0.9999, warmup=False):
        if not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError("EMA decay must be finite and in [0, 1]")
        self.model, self.decay, self.warmup = model, decay, warmup
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in self._parameters().items()}
        self.backup = {}
        self._applied = False

    def _parameters(self):
        return {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def _effective_decay(self):
        return min(self.decay, (1 + self.updates) / (10 + self.updates)) if self.warmup else self.decay

    @torch.no_grad()
    def update(self):
        if self._applied:
            raise RuntimeError("Cannot update EMA while its shadow is applied")
        self.updates += 1
        decay = self._effective_decay()
        for name, parameter in self._parameters().items():
            # Preserve the native trainer's arithmetic order.
            self.shadow[name] = decay * self.shadow[name] + (1 - decay) * parameter

    @torch.no_grad()
    def apply_shadow(self):
        if self._applied:
            raise RuntimeError("EMA shadow is already applied")
        parameters = self._parameters()
        self.backup = {n: p.detach().clone() for n, p in parameters.items()}
        for name, parameter in parameters.items():
            parameter.copy_(self.shadow[name])
        self._applied = True

    @torch.no_grad()
    def restore(self):
        if not self._applied:
            raise RuntimeError("EMA shadow has not been applied")
        for name, parameter in self._parameters().items():
            parameter.copy_(self.backup[name])
        self.backup = {}
        self._applied = False

    @contextmanager
    def average_parameters(self):
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self):
        return self.shadow.copy()

    def metadata(self):
        return dict(version=1, updates=self.updates, decay=self.decay, warmup=self.warmup)

    def load_state_dict(self, state_dict, *, metadata=None):
        """Validate before mutation; clone/cast weights onto the model's device.

        Legacy checkpoints lack the update count. Only warmup EMA needs a mature
        fallback, and this limitation is explicitly warned rather than hidden.
        """
        if self._applied:
            raise RuntimeError("Cannot load EMA while its shadow is applied")
        parameters, updates, decay, warmup = validate_ema_state(
            self.model, state_dict, metadata=metadata, decay=self.decay, warmup=self.warmup)
        if metadata is None and warmup:
            warnings.warn("Legacy checkpoint has no EMA update count; using mature decay. "
                          "Exact warmup continuation is unavailable.", UserWarning, stacklevel=2)
        shadow = {n: state_dict[n].detach().to(device=p.device, dtype=p.dtype).clone()
                  for n, p in parameters.items()}
        self.shadow = shadow
        self.updates, self.decay, self.warmup = updates, decay, warmup
