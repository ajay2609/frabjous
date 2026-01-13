"""
Type-B FRB simulation script.
"""

from simpulse import single_pulse as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import yaml
import random
import os

from utils import (
    gaussian_filter,
    calculate_noise_rms,
    write_out_frbs,
)

##global parameters to simulate type B bursts

PULSE_NT = 512
NFREQ = 256
FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0

CROP_CENTER = 215
CROP_HALF_WIDTH = 128


# ----------------------------------
# functions to simulate
# ----------------------------------

def generate_frb(
    frb_dict,
    central_frequency,
    bandwidth,
    snr,
    add_noise=True,
):
    """
    Generate a simulated Type-B FRB dynamic spectrum.

    Parameters
    ----------
    frb_dict : dict
        FRB parameters.
    central_frequency : float
        Central frequency of the Gaussian band.
    bandwidth : float
        Bandwidth of the emission.
    snr : float
        Target signal-to-noise ratio.
    add_noise : bool
        Add Gaussian noise if True.

    Returns
    -------
    np.ndarray
        Dynamic spectrum (frequency × time).
    """

    ts = np.zeros((NFREQ, PULSE_NT))

    for comp in frb_dict["components"]:

        frb_pulse = sp(
            PULSE_NT,
            NFREQ,
            FREQ_LO_MHZ,
            FREQ_HI_MHZ,
            frb_dict["dm"],
            frb_dict["sm"],
            comp["width"],
            comp["fluence"],
            comp.get("spectral_index", 0.0),
            comp["arrival_time"],
        )

        comp_ts = np.zeros_like(ts)
        frb_pulse.add_to_timestream(comp_ts, 2.8, 3.3)

        # Narrow-band spectral structure
        obs_freq = [FREQ_LO_MHZ, FREQ_HI_MHZ]
        comp_ts = gaussian_filter(
            comp_ts,
            comp_ts.shape,
            central_frequency,
            bandwidth,
            obs_freq,
        )

        ts += comp_ts

    ts /= 100.0

    noise_rms = calculate_noise_rms(ts, snr)

    if add_noise:
        ts += np.random.normal(0.0, noise_rms, ts.shape)

    ts = np.flip(ts, axis=0)

    s = CROP_CENTER - CROP_HALF_WIDTH
    e = CROP_CENTER + CROP_HALF_WIDTH

    return ts[:, s:e]




def simulate_frbs(config_file):
    """
    Simulate a population of Type-B FRBs.
    """

    with open(config_file) as f:
        params = yaml.safe_load(f)

    count = params["count"]
    snr = params.get("snr", 50.0)
    add_noise = params.get("add_noise", True)

    dm_vals = np.linspace(
        params["dm_range"]["lower_dm_value"],
        params["dm_range"]["upper_dm_value"],
        count,
    )

    sm_vals = np.linspace(
        params["scattering_measure"]["sm_lower"],
        params["scattering_measure"]["sm_upper"],
        count,
    )

    component_0 = params["component_0"]

    # --- Width distribution ---
    width = np.linspace(
        component_0["width"]["width_lower_value"],
        component_0["width"]["width_upper_value"],
        int(count * 0.2),
    )

    lognorm_widths = lognorm.rvs(
        0.9311,
        loc=0,
        scale=0.0011,
        size=int(count * 0.8),
    )

    lognorm_widths[lognorm_widths < 0.0001] = 0.0001
    width = np.concatenate((width, lognorm_widths))
    # -------------------------

    fluence = np.linspace(
        component_0["fluence"]["fluence_lower_value"],
        component_0["fluence"]["fluence_upper_value"],
        count,
    )

    spectral_index = np.linspace(
        component_0["spectral_index"]["spec_lower_value"],
        component_0["spectral_index"]["spec_upper_value"],
        count,
    )

    spectral_running = np.linspace(
        component_0["spectral_running"]["sr_lower_value"],
        component_0["spectral_running"]["sr_upper_value"],
        count,
    )

    central_freq = np.linspace(
        params["central_freq"]["lower_value"],
        params["central_freq"]["upper_value"],
        count,
    )

    bandwidth = np.linspace(
        params["bandwidth"]["lower_value"],
        params["bandwidth"]["upper_value"],
        count,
    )

    random.shuffle(dm_vals)
    random.shuffle(sm_vals)
    random.shuffle(width)
    random.shuffle(fluence)
    random.shuffle(spectral_index)
    random.shuffle(spectral_running)
    random.shuffle(central_freq)
    random.shuffle(bandwidth)

    data = []
    frb_metadata = []
    min_max = []

    for i in range(count):

        frb_dict = {
            "id": f"frb_{i}",
            "dm": dm_vals[i],
            "sm": sm_vals[i],
            "components": [
                {
                    "arrival_time": component_0["arrival_time"],
                    "width": width[i],
                    "fluence": fluence[i],
                    "spectral_index": -spectral_index[i],
                    "spectral_running": spectral_running[i],
                }
            ],
        }

        img = generate_frb(
            frb_dict,
            central_freq[i],
            bandwidth[i],
            snr,
            add_noise,
        )

        data.append(img)
        frb_metadata.append(frb_dict)
        min_max.append((img.max(), img.min()))

    return data, frb_metadata, min_max




if __name__ == "__main__":

    CONFIG_FILE = "generate_FRBs_config_type_B.yml"

    data, frb_header, min_max = simulate_frbs(CONFIG_FILE)

    write_out_frbs(data, frb_header, min_max, "B")

    os.makedirs("type_B_images", exist_ok=True)

    for i, img in enumerate(data[:1000]):

        frb = frb_header[i]

        plt.imshow(img, aspect="auto")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title(
            f"Width={frb['components'][0]['width']:.4f}, "
            f"Fluence={frb['components'][0]['fluence']:.2f}"
        )
        plt.colorbar()
        plt.savefig(f"type_B_images/frb_type_B_{i}.png", dpi=150)
        plt.clf()

