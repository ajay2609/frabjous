"""
Type-D FRB simulation script.

This script generates a population of single-component FRBs
which are detected in the sidelobe of a radio telescope
"""

from simpulse import single_pulse as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import yaml
import random
import time

from utils import (
    apply_spectral_index_and_running,
    calculate_noise_rms,
    write_out_frbs,
    fraunhoffer_pattern,
)

### global parameters for generating dynamic spectra #### 

PULSE_NT = 512
NFREQ = 256

FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0

CROP_CENTER = 215
CROP_HALF_WIDTH = 128




class CreateFRBTypeD:
    """
    Generate a population of Type-D FRBs.
    """

    def __init__(
        self,
        config_file: str = "config_type_VI.yml",
        chime_catalog: str = "chimefrbcat1.csv",
    ):
        """
        Parameters
        ----------
        config_file : str
            YAML configuration file defining parameter ranges.
        chime_catalog : str
            CHIME/FRB catalog used for spectral index distributions.
        """

        with open(config_file) as f:
            self.input_params = yaml.safe_load(f)

        self.csvfile = pd.read_csv(chime_catalog)

        self.noise = self.input_params["add_noise"]
        self.count = self.input_params["count"]

        self.dm_range = self.input_params["dm_range"]
        self.scattering_measure = self.input_params["scattering_measure"]
        self.component_0 = self.input_params["component_0"]
        self.theta_range = self.input_params["theta"]

    # --------------------------------------------------------

    def simulate_frbs(self, snr: float = 20.0):
        """
        Generate a population of Type-D FRBs.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio.

        Returns
        -------
        data : list of ndarray
            Simulated dynamic spectra.
        frb_metadata : list of dict
            Metadata for each FRB.
        min_max : list of tuple
            (max, min) values for each dynamic spectrum.
        """

        DM = np.linspace(
            self.dm_range["lower_dm_value"],
            self.dm_range["upper_dm_value"],
            self.count,
        )

        SM = np.linspace(
            self.scattering_measure["sm_lower"],
            self.scattering_measure["sm_upper"],
            self.count,
        )

        width = np.linspace(
            self.component_0["width"]["width_lower_value"],
            self.component_0["width"]["width_upper_value"],
            self.count,
        )

        fluence = np.linspace(
            self.component_0["fluence"]["fluence_lower_value"],
            self.component_0["fluence"]["fluence_upper_value"],
            self.count,
        )

        spectral_index = np.linspace(
            self.component_0["spectral_index"]["spec_lower_value"],
            self.component_0["spectral_index"]["spec_upper_value"],
            self.count,
        )

        thetas = np.linspace(
            self.theta_range["min"],
            self.theta_range["max"],
            self.count,
        )

        arrival_time = self.component_0["arrival_time"]

        # Shuffle parameters
        random.shuffle(DM)
        random.shuffle(SM)
        random.shuffle(width)
        random.shuffle(fluence)
        random.shuffle(spectral_index)
        random.shuffle(thetas)

        data = []
        frb_metadata = []
        min_max = []

        for i in range(self.count):

            frb_dict = {
                "id": f"frb_{i}",
                "dm": DM[i],
                "sm": SM[i],
                "theta": thetas[i],
                "components": [
                    {
                        "arrival_time": arrival_time,
                        "width": width[i],
                        "fluence": fluence[i],
                        "spectral_index": spectral_index[i],
                        "spectral_running": 0.0,
                    }
                ],
            }

            img = self.generate_frb(frb_dict, snr)

            data.append(img)
            frb_metadata.append(frb_dict)
            min_max.append((img.max(), img.min()))

        return data, frb_metadata, min_max

    # --------------------------------------------------------

    def generate_frb(self, frb_dict: dict, snr: float):
        """
        Generate a single Type-D FRB dynamic spectrum.
        """

        waterfall = np.zeros((NFREQ, PULSE_NT))

        comp = frb_dict["components"][0]

        frb_pulse = sp(
            PULSE_NT,
            NFREQ,
            FREQ_LO_MHZ,
            FREQ_HI_MHZ,
            frb_dict["dm"],
            frb_dict["sm"],
            comp["width"],
            comp["fluence"],
            0.0,
            comp["arrival_time"],
        )

        ts = np.zeros((NFREQ, PULSE_NT))
        frb_pulse.add_to_timestream(ts, 2.8, 3.3)

        ts = apply_spectral_index_and_running(
            ts,
            ts.shape,
            comp["spectral_index"],
            comp["spectral_running"],
        )

        ts = fraunhoffer_pattern(ts, frb_dict["theta"])

        waterfall += ts
        waterfall /= 100.0

        noise_rms = calculate_noise_rms(waterfall, snr)

        if self.noise:
            waterfall += np.random.normal(0, noise_rms, waterfall.shape)

        s = CROP_CENTER - CROP_HALF_WIDTH
        e = CROP_CENTER + CROP_HALF_WIDTH

        return waterfall[:, s:e]


######## generate FRB samples ##############

if __name__ == "__main__":

    start = time.time()

    simulator = CreateFRBTypeD()
    data, frb_header, min_max = simulator.simulate_frbs(snr=50.0)

    write_out_frbs(data, frb_header, min_max, "D")

    print(f"Type-D simulation completed in {time.time() - start:.1f} s")

    # --------------------------------------------------------
    # Optional: save example images for inspection
    # --------------------------------------------------------
    # import matplotlib.pyplot as plt
    # for i, img in enumerate(data[:10]):
    #     plt.imshow(img, aspect="auto", origin="lower")
    #     plt.colorbar()
    #     plt.savefig(f"type_D_images/frb_type_D_{i}.png", dpi=150)
    #     plt.clf()

