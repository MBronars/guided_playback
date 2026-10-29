"""State-dict loading helpers for legacy checkpoints."""
from typing import Dict

import torch


def strip_contact_wrapper(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map legacy contact-wrapped checkpoint keys onto the bare backbone layout.

    Pre-no-contact branch, `TransformerWithContact` saved keys as `backbone.<name>`
    alongside `contact_head.*`. Here we strip `backbone.` and drop contact keys so
    those checkpoints still load (the contact head is gone on this branch).
    Already-bare state dicts pass through unchanged.
    """
    if not any(k.startswith('backbone.') for k in state_dict):
        return state_dict
    return {k[len('backbone.'):] if k.startswith('backbone.') else k: v
            for k, v in state_dict.items() if not k.startswith('contact_head.')}


def load_model_state(model, state_dict):
    """Validate before mutation; EMA may omit only regenerated embedding buffers.

    A missing learned parameter, an unknown key, or a wrong tensor shape is
    fatal. Do not label missing learned weights as 'expected for buffers'.
    """
    state = strip_contact_wrapper(state_dict)
    expected = model.state_dict()
    regenerated = {name for name, _ in model.named_buffers()
                   if name.endswith((".freqs", ".pe"))}
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    wrong_shapes = [name for name in set(expected) & set(state)
                    if expected[name].shape != state[name].shape]
    errors = []
    if missing - regenerated:
        errors.append(f"Missing learned/state keys: {sorted(missing - regenerated)}")
    if unexpected:
        errors.append(f"Unexpected keys: {sorted(unexpected)}")
    if wrong_shapes:
        errors.append(f"Wrong tensor shapes: {sorted(wrong_shapes)}")
    if errors:
        raise RuntimeError("Checkpoint is incompatible: " + "; ".join(errors))
    model.load_state_dict(state, strict=not missing)
    return sorted(missing)
