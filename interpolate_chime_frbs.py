"""
Interpolate masked frequency channels in CHIME/FRB waterfall data.

This module:
- Loads CHIME burst HDF5 files
- Masks RFI-contaminated frequency channels
- Interpolates missing frequency bins
- Produces fixed-size waterfall images suitable for ML applications

Author: Ajay Kumar
"""

from __future__ import annotations

import os
import glob
import argparse
import warnings
from typing import Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ==============================
# Utility functions=
#===============================

def bin_freq_channels(data: np.ndarray, fbin_factor: int) -> np.ndarray:
    """
    Bin frequency channels by averaging.

    Parameters
    ----------
    data : np.ndarray
        Waterfall array with shape (nchan, nt).
    fbin_factor : int
        Number of channels to bin.

    Returns
    -------
    np.ndarray
        Averaged waterfall.
    """
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        raise ValueError("fbin_factor must divide number of channels")

    reshaped = data.reshape(num_chan // fbin_factor, fbin_factor, data.shape[1])
    return np.nanmean(reshaped, axis=1)


def image_interpolation(img: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values along frequency axis for each time bin with linear interpolation.
    """
    n_fsamp, n_tsamp = img.shape
    indices = np.arange(n_fsamp)

    out = img.copy()
    for t in range(n_tsamp):
        valid = np.isfinite(out[:, t])
        if np.sum(valid) < 2:
            continue
        f = interp1d(indices[valid], out[:, t][valid], bounds_error=False)
        out[:, t] = np.where(valid, out[:, t], f(indices))

    return np.nan_to_num(out)


def determine_snr(model_ts: np.ndarray, rms: float) -> float:
    """
    Compute SNR using cumulative-energy effective width.
    """
    cumsum = np.cumsum(model_ts)
    t95 = np.where(cumsum > 0.95 * cumsum.max())[0][0]
    t05 = np.where(cumsum > 0.05 * cumsum.max())[0][0]
    width = max(t95 - t05, 1)
    return np.sum(model_ts) / (np.sqrt(width) * rms)


def replace_missing_values_median(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Replace NaNs in 1D array using median-filtered local sampling.
    """
    out = data.copy()
    nan_idx = np.where(np.isnan(out))[0]

    if nan_idx.size == 0:
        return out

    splits = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)

    for grp in splits:
        x0 = grp[0]
        q = max(len(grp) * 3 // 2, 10)
        left = np.where(~np.isnan(out[:x0]))[0][-q:]
        right = np.where(~np.isnan(out[x0:]))[0][:q] + x0
        valid = np.concatenate([left, right])

        if valid.size == 0:
            continue

        vals = out[valid]
        med = np.median(vals)
        mad = median_abs_deviation(vals) * 2
        vals = vals[(vals > med - mad) & (vals < med + mad)]

        if vals.size == 0:
            continue

        for x in grp:
            out[x] = rng.choice(vals)

    return out


def interpolate_waterfall(wfall: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Interpolate masked channels and estimate noise statistics.
    """
    image = wfall.copy()
    ts = np.nansum(image, axis=0)
    noise_idx = np.where(ts < np.std(ts))[0]

    for t in range(image.shape[1]):
        image[:, t] = replace_missing_values_median(image[:, t], rng)

    image = bin_freq_channels(image, 4)

    mean = np.nanmean(image[:, noise_idx])
    std = np.nanstd(image[:, noise_idx])

    return image, mean, std, noise_idx


# Notes on fixed methodological choices used below:
# - Frequency binning: we first go from 16384 channels to 1024 channels for interpolating
# - Output image size: 256 × 256 (time × frequency)
# - RFI masking: channels with variance > 3 × mean channel variance
# - Noise estimation from off-pulse time bins

def reshape_burst(h5file: h5py.File, rng: np.random.Generator, visualize: bool = False):
    """
    Process a single CHIME burst HDF5 file into fixed-size images.
    First it masks the channles with RFI
    Interpolating the masked frequency channels.

    Input : raw burst data with 16384 freqeuncy channels and arbitray time samples
    Returns: interpolated 2-D numoy array (Dyanmic spectra) with 256x256 pixels
    """
    data = h5file["frb"]

    wfall = data["wfall"][:]
    model_wfall = data["model_wfall"][:]
    ts = data["ts"][:]
    model_ts = data["model_ts"][:]
    spec = data["spec"][:]

    # RFI masking
    chan_var = np.nanvar(wfall, axis=1)
    mean_var = np.nanmean(chan_var)
    q1, q3 = np.nanquantile(spec, [0.25, 0.75])
    iqr = q3 - q1

    with np.errstate(invalid="ignore"):
        rfi_mask = (
            (chan_var > 3.0 * mean_var)  # RFI variance threshold
            | (spec[::-1] < q1 - 1.5 * iqr)
            | (spec[::-1] > q3 + 1.5 * iqr)
        )

    wfall[rfi_mask] = np.nan
    model_wfall[rfi_mask] = np.nan

    # Bin and interpolate
    wfall = bin_freq_channels(wfall, 16)  # frequency binning factor
    interp_img, mean, std, noise_idx = interpolate_waterfall(wfall, rng)

    rms = np.sqrt(np.mean(ts[noise_idx] ** 2))
    model_snr = determine_snr(model_ts, rms)

    model_wfall = bin_freq_channels(model_wfall, 64)  # model frequency binning factor
    try:
        model_wfall = image_interpolation(model_wfall)
    except Exception as e:
        warnings.warn(f"interpolation failed: {e}")
        model_wfall = np.nan_to_num(model_wfall)

    # Center / crop
    img = rng.normal(mean, std, (256, 256))  # output image size
    model_img = np.zeros((256, 256))

    nt = interp_img.shape[1]
    if nt >= 256:
        img = interp_img[:, :256]
        model_img = model_wfall[:, :256]
    else:
        start = 256 // 2 - nt // 2
        img[:, start:start + nt] = interp_img
        model_img[:, start:start + nt] = model_wfall

    if visualize:
        plt.imshow(img, aspect="auto")
        plt.show()

    return img, model_img, model_snr


# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="Interpolate CHIME FRB waterfalls")
    parser.add_argument("--input-dir", required=True, help="Directory containing CHIME HDF5 files")
    parser.add_argument("--output", default="chime_interp", help="Output file prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    
    bursts = []
    model_bursts = []
    msnrs = []

    for i, fname in enumerate(files, start=1):
        with h5py.File(fname, "r") as f:
            img, model_img, msnr = reshape_burst(f, rng)
            bursts.append(img)
            model_bursts.append(model_img)
            msnrs.append(msnr)
            logger.info(f"Processed {i} bursts")

    np.savez_compressed(f"{args.output}_data.npz", np.array(bursts))
    np.savez_compressed(f"{args.output}_model.npz", np.array(model_bursts))
    np.savetxt(f"{args.output}_msnr.txt", msnrs)


if __name__ == "__main__":
    main()

