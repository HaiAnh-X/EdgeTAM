import time
import torch
from .hosvd import hosvd as hosvd_classic, restore_hosvd
from .hosvd_subspace_iteration import hosvd_subspace_iteration, restore_hosvd as restore_hosvd_subspace_iteration
from sam2.modeling.sam2_base import SAM2Base

# --- Global state ---
_previous_Ulist = None

stats = {
    "hosvd_overhead": 0.0,
    "hosvd_memory":   [],
    "hosvd_mse":      [],
    "hosvd_subspace_iteration_overhead": 0.0,
    "hosvd_subspace_iteration_memory":   [],
    "hosvd_subspace_iteration_mse":      [],
}

# --- PATCH ---
def apply_patch(mode, rank=48, var_threshold=0.99):
    global _previous_Ulist
    _previous_Ulist = None

    if not hasattr(SAM2Base, '_orig_encode'):
        SAM2Base._orig_encode = SAM2Base._encode_new_memory

    if mode == "baseline":
        SAM2Base._encode_new_memory = SAM2Base._orig_encode
        return

    if mode == "hosvd":
        def hook(self, *args, **kwargs):
            feat, pos = self._orig_encode(*args, **kwargs)
            fd = feat.detach().float()
            t0 = time.perf_counter()
            S, ul = hosvd_classic(fd, var=var_threshold)
            restored = restore_hosvd(S, ul)
            stats["hosvd_overhead"] += time.perf_counter() - t0
            stats["hosvd_memory"].append((
                fd.numel() * fd.element_size(),
                S.numel() * S.element_size() + sum(u.numel() * u.element_size() for u in ul)
            ))
            stats["hosvd_mse"].append(torch.mean((fd - restored) ** 2).item())
            return restored.to(dtype=feat.dtype, device=feat.device), pos

    elif mode == "hosvd_subspace_iteration":
        def hook(self, *args, **kwargs):
            global _previous_Ulist
            feat, pos = self._orig_encode(*args, **kwargs)
            fd = feat.detach().float()

            can_reuse = (
                _previous_Ulist is not None
                and len(_previous_Ulist) == fd.dim()
                and all(u.shape[0] == fd.shape[i] for i, u in enumerate(_previous_Ulist))
            )

            t0 = time.perf_counter()
            S, ul = hosvd_subspace_iteration(fd, _previous_Ulist, reuse_U=can_reuse, rank=rank)
            _previous_Ulist = ul
            restored = restore_hosvd_subspace_iteration(S, ul)
            stats["hosvd_subspace_iteration_overhead"] += time.perf_counter() - t0
            stats["hosvd_subspace_iteration_memory"].append((
                fd.numel() * fd.element_size(),
                S.numel() * S.element_size() + sum(u.numel() * u.element_size() for u in ul)
            ))
            stats["hosvd_subspace_iteration_mse"].append(torch.mean((fd - restored) ** 2).item())
            return restored.to(dtype=feat.dtype, device=feat.device), pos

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: baseline | hosvd | hosvd_subspace_iteration")

    SAM2Base._encode_new_memory = hook
