#!/usr/bin/env python3
import cv2
import numpy as np
import math

# ================= USER CONFIG ================= #
INPUT_PATH = "D:\\SyntheticData\\Unreal SDG\\spalling\\C13\\MetroTrack.0213.jpeg"
OUTPUT_PATH = "D:\\SyntheticData\\Unreal SDG\\spalling\\C13\\noise\\image_noise.jpg"

# Column Fixed-Pattern Noise (CFPN)
COLUMN_OFFSET_RANGE = 6.0  # max absolute offset per column (in pixel values)
COLUMN_GROUP_WIDTH = 1     # group adjacent columns into wider bands

# Low-frequency column-wise bias drift
BIAS_STRENGTH = 4.0        # amplitude of smoothed random drift (pixel values)
BIAS_SMOOTH_SIGMA = 30.0   # smoothness in columns (Gaussian sigma)
BIAS_SINE_AMPLITUDE = 0.0  # optional very low-frequency sine amplitude
BIAS_SINE_PERIOD = 300.0   # period in columns for sine drift

# Pixel-wise Gaussian noise
PIXEL_NOISE_SIGMA = 2.0    # per-pixel Gaussian noise sigma (pixel values)

# Other
APPLY_TO_LUMA = True       # for RGB, apply noise to luminance (Y) only
RANDOM_SEED = None         # set int for reproducibility
# =============================================== #


def gaussian_kernel1d(sigma):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def smooth_1d(signal, sigma):
    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return signal
    return np.convolve(signal, kernel, mode="same")


def build_column_offsets(width, rng, max_abs, group_width):
    group_width = max(1, int(group_width))
    groups = int(math.ceil(width / group_width))
    offsets = rng.uniform(-max_abs, max_abs, size=groups).astype(np.float32)
    offsets = np.repeat(offsets, group_width)[:width]
    return offsets


def build_bias_drift(width, rng, strength, smooth_sigma, sine_amp, sine_period):
    drift = np.zeros(width, dtype=np.float32)
    if strength > 0:
        noise = rng.normal(0.0, 1.0, size=width).astype(np.float32)
        smooth = smooth_1d(noise, smooth_sigma)
        if np.max(np.abs(smooth)) > 0:
            smooth = smooth / np.max(np.abs(smooth))
        drift += smooth * strength
    if sine_amp > 0 and sine_period > 0:
        phase = rng.uniform(0.0, 2.0 * math.pi)
        x = np.arange(width, dtype=np.float32)
        sine = np.sin(2.0 * math.pi * (x / sine_period) + phase)
        drift += sine * sine_amp
    return drift


def apply_cfpn_to_luma(luma, rng, pixel_sigma):
    h, w = luma.shape
    col_offsets = build_column_offsets(w, rng, COLUMN_OFFSET_RANGE, COLUMN_GROUP_WIDTH)
    bias = build_bias_drift(w, rng, BIAS_STRENGTH, BIAS_SMOOTH_SIGMA,
                            BIAS_SINE_AMPLITUDE, BIAS_SINE_PERIOD)
    per_col = (col_offsets + bias).reshape(1, w)
    out = luma + per_col
    if pixel_sigma > 0:
        out += rng.normal(0.0, pixel_sigma, size=luma.shape).astype(np.float32)
    return out


def clamp_to_dtype(arr, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    img = cv2.imread(INPUT_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(INPUT_PATH)

    if img.ndim == 2:
        luma = img.astype(np.float32)
        noisy = apply_cfpn_to_luma(luma, rng, PIXEL_NOISE_SIGMA)
        out = clamp_to_dtype(noisy, img.dtype)
    else:
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3:]
        else:
            bgr = img
            alpha = None

        if APPLY_TO_LUMA:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[:, :, 0].astype(np.float32)
            y_noisy = apply_cfpn_to_luma(y, rng, PIXEL_NOISE_SIGMA)
            ycrcb[:, :, 0] = clamp_to_dtype(y_noisy, ycrcb.dtype)
            out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            bgr_f = bgr.astype(np.float32)
            noisy = apply_cfpn_to_luma(bgr_f.mean(axis=2), rng, 0.0)
            per_col = (noisy - bgr_f.mean(axis=2)).reshape(bgr_f.shape[0], bgr_f.shape[1], 1)
            out_bgr = bgr_f + per_col
            if PIXEL_NOISE_SIGMA > 0:
                out_bgr += rng.normal(0.0, PIXEL_NOISE_SIGMA, size=bgr_f.shape).astype(np.float32)
            out_bgr = clamp_to_dtype(out_bgr, bgr.dtype)

        if alpha is not None:
            out = np.concatenate([out_bgr, alpha], axis=2)
        else:
            out = out_bgr

    cv2.imwrite(OUTPUT_PATH, out)


if __name__ == "__main__":
    main()
