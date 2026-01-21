"""
Type-IV FRB simulation script.
"""

from simpulse import single_pulse as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
import yaml
import random
import time

from utils import (
    apply_spectral_index_and_running,
    calculate_noise_rms,
    write_out_frbs,
)

####global parameters for generating dynamic spectra for all type IV bursts

PULSE_NT = 512
NFREQ = 256
FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0

CROP_CENTER = 230
CROP_HALF_WIDTH = 128


class CreateFRB:
    """
    Generate a population of Type-C1 FRBs with multiple components.
    """

    def __init__(
        self,
        config_file: str = "config_type_IV.yml",
        chime_catalog: str = "chimefrbcat1.csv",
    ):
        with open(config_file) as f:
            self.input_params = yaml.safe_load(f)

        self.csvfile = pd.read_csv(chime_catalog)
        self.snr = self.input_params["snr"]

        self.noise = self.input_params["add_noise"]
        self.dm_range = self.input_params["dm_range"]
        self.scattering_measure = self.input_params["scattering_measure"]
        self.component_0 = self.input_params["component_0"]
        self.count = self.input_params["count"]

        self.dm_lo = self.dm_range["lower_dm_value"]
        self.dm_up = self.dm_range["upper_dm_value"]

        # spectral index / running pairs from CHIME/FRB catalog
        self.spectra = list(
            zip(self.csvfile["sp_idx"].values, self.csvfile["sp_run"].values)
        )

    # -------------------------------------------------

    def simulate_frbs(self, snr: float = 10.0):
        """
        Generate a population of Type-C1 FRBs.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio.

        Returns
        -------
        data, frb_metadata, min_max
        """

        DM = np.linspace(self.dm_lo, self.dm_up, self.count)
        SM = np.linspace(
            self.scattering_measure["sm_lower"],
            self.scattering_measure["sm_upper"],
            self.count,
        )

        arrival_time = self.component_0["arrival_time"]

        
        width = np.linspace(
            self.component_0["width"]["width_lower_value"],
            self.component_0["width"]["width_upper_value"],
            int(self.count * 0.2),
        )
	
	# Width distribution from the first CHIME/FRB catalog ---
        lognorm_widths = lognorm.rvs(
            0.9311,
            loc=0,
            scale=0.0011,
            size=int(self.count * 0.8),
        )
        lognorm_widths[lognorm_widths < 0.0001] = 0.0001
        width = np.concatenate((width, lognorm_widths))
        
        # ------------------------------------------------
        fluence = np.linspace(
            self.component_0["fluence"]["fluence_lower_value"],
            self.component_0["fluence"]["fluence_upper_value"],
            self.count,
        )

        arrival_time_sep = np.linspace(0.005, 0.025, self.count)
        fluence_ratio = np.linspace(0.2, 1.5, self.count)

        num_components_choices = [2, 3]

        # Shuffle parameters
        random.shuffle(DM)
        random.shuffle(SM)
        random.shuffle(width)
        random.shuffle(fluence)
        random.shuffle(arrival_time_sep)
        random.shuffle(fluence_ratio)

        data = []
        frb_metadata = []
        min_max = []

        for i in range(self.count):

            spec_idx, spec_run = random.choice(self.spectra)
            ncomp = random.choice(num_components_choices)

            components = [
                {
                    "arrival_time": arrival_time,
                    "width": width[i],
                    "fluence": fluence[i],
                    "spectral_index": spec_idx,
                    "spectral_running": spec_run,
                }
            ]

            if ncomp >= 2:
                components.append(
                    {
                        "arrival_time": arrival_time + arrival_time_sep[i],
                        "width": self._choose_width(width, arrival_time_sep[i]),
                        "fluence": fluence[i] * random.choice(fluence_ratio),
                        "spectral_index": spec_idx,
                        "spectral_running": spec_run,
                    }
                )

            if ncomp == 3:
                components.append(
                    {
                        "arrival_time": arrival_time
                        + arrival_time_sep[i]
                        + arrival_time_sep[i - 1],
                        "width": self._choose_width(width, arrival_time_sep[i - 1]),
                        "fluence": fluence[i] * random.choice(fluence_ratio),
                        "spectral_index": spec_idx,
                        "spectral_running": spec_run,
                    }
                )

            frb_dict = {
                "id": f"frb_{i}",
                "dm": DM[i],
                "sm": SM[i],
                "components": components,
                "arrival_time_sep": arrival_time_sep[i],
            }

            img = self.generate_frb(frb_dict, ncomp, snr)

            data.append(img)
            frb_metadata.append(frb_dict)
            min_max.append((img.max(), img.min()))

        return data, frb_metadata, min_max

    # -------------------------------------------------

    def generate_frb(self, frb_dict, ncomp, snr):
        """
        Generate a single Type-C1 FRB dynamic spectrum.
        """

        waterfall = np.zeros((NFREQ, PULSE_NT))

        for i in range(ncomp):

            comp = frb_dict["components"][i]

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

            waterfall += ts

        waterfall /= 100.0

        noise_rms = calculate_noise_rms(waterfall, snr)

        if self.noise:
            waterfall += np.random.normal(0, noise_rms, waterfall.shape)

        waterfall = np.flip(waterfall, axis=0)

        s = CROP_CENTER - CROP_HALF_WIDTH
        e = CROP_CENTER + CROP_HALF_WIDTH

        return waterfall[:, s:e]

    # -------------------------------------------------

    def _choose_width(self, width_array, separation):
        """
        Choose a width smaller than half the component separation.
        """
        valid = [w for w in width_array if w < (separation / 2)]
        return random.choice(valid)



if __name__ == "__main__":

    start = time.time()

    simulator = CreateFRB()
    data, frb_header, min_max = simulator.simulate_frbs(snr=50.0)

    write_out_frbs(data, frb_header, min_max, "C1")
    
    print(f"Simulation completed in {time.time() - start:.1f} s")

## comment out this part if you want to save and check the generated samples in png format

#    for i, img in enumerate(data[:]):
#        plt.imshow(img, aspect="auto")
#        plt.colorbar()
#        plt.savefig(f"type_C1_images/frb_type_C1_{i}.png", dpi=150)
#        plt.clf()

    

