import numpy as np


def _safe_scale_and_clip(pks, from_sr, to_sr, L):
    if pks is None:
        return np.array([], dtype=int)
    a = np.asarray(pks)
    if a.size == 0:
        return np.array([], dtype=int)
    # binary/boolean vector matching target length -> indices
    if a.ndim == 1 and a.size == L and set(np.unique(a)).issubset({0, 1, True, False}):
        idx = np.where(a == 1)[0]
    else:
        # scale then floor to avoid producing index == L by rounding up
        scaled = np.floor(a.astype(float) * (to_sr / float(from_sr))).astype(int)
        idx = scaled
    idx = np.unique(idx[(idx >= 0) & (idx <= L - 1)])
    return idx.astype(int)


def _sanitize_peaks(peaks, sig_len):
    if peaks is None:
        return np.array([], dtype=int)
    # handle many return types robustly
    if isinstance(peaks, dict) and "ECG_R_Peaks" in peaks:
        arr = np.asarray(peaks["ECG_R_Peaks"])
        return _sanitize_peaks(arr, sig_len)
    a = np.asarray(peaks)
    if a.size == 0:
        return np.array([], dtype=int)
    # boolean/binary vector -> indices (truncate if longer)
    if a.ndim == 1 and set(np.unique(a)).issubset({0, 1, True, False}):
        if a.size > sig_len:
            a = a[:sig_len]
        idx = np.where(a == 1)[0]
    else:
        # floats/ints -> floor then clip
        idx = np.floor(a.astype(float)).astype(int)
    idx = np.unique(idx[(idx >= 0) & (idx <= sig_len - 1)])
    return idx.astype(int)


def _round_and_clip_indices(pks, L, sig=None, sig_name="signal"):
    """
    Robust wrapper that rounds/clips/sanitizes detector outputs to valid int indices.
    Produces diagnostic plot on error and returns a clipped fallback.
    """
    try:
        out = _sanitize_peaks(pks, L)
        if out.size == 0:
            return out
        if np.any(out < 0) or np.any(out >= L) or np.isnan(out).any():
            raise ValueError("sanitized indices out-of-bounds or NaN")
        return out
    except Exception as e:
        print(f"_round_and_clip_indices warning for {sig_name}: {e}")
        # fallback: try best-effort rounding+clipping
        try:
            a = np.asarray(pks)
            if a.size == 0:
                fallback = np.array([], dtype=int)
            elif (
                    a.ndim == 1
                    and a.size == L
                    and set(np.unique(a)).issubset({0, 1, True, False})
                ):
                fallback = np.where(a == 1)[0]
            else:
                fallback = np.round(a.astype(float)).astype(int)
                fallback = np.unique(np.clip(fallback, 0, L - 1))
        except Exception:
            fallback = np.array([], dtype=int)
        return fallback.astype(int)
