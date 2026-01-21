"""
Type-III FRB simulation script.
"""

from simpulse import single_pulse as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import yaml
import random
import time

from utils import (
    gaussian_filter,
    calculate_noise_rms,
    write_out_frbs,
)


######### Global paramters for simulating all the bursts #############

PULSE_NT = 512
NFREQ = 256
FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0

CROP_CENTER = 215
CROP_HALF_WIDTH = 128



class CreateFRB:
    """
    class to generate a sample of type III bursts
    """

    def __init__(self, config_file: str = "config_type_III.yml"):
        with open(config_file) as f:
            self.input_params = yaml.safe_load(f)
        self.snr = self.input_params["snr"]
        self.noise = self.input_params["add_noise"]
        self.dm_range = self.input_params["dm_range"]
        self.scattering_measure = self.input_params["scattering_measure"]
        self.component_0 = self.input_params["component_0"]
        self.count = self.input_params["count"]
        self.peaks = self.input_params["peaks"]

        self.dm_lo = self.dm_range["lower_dm_value"]
        self.dm_up = self.dm_range["upper_dm_value"]

    # -------------------------------------------------

    def simulate_frbs(self, snr: float = 10.0):
        """
        Generate a population of Type-C FRBs.

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

        # --------Width distribution from CHIME/FRB first catalog -----------------------------------
        width = np.linspace(
            self.component_0["width"]["width_lower_value"],
            self.component_0["width"]["width_upper_value"],
            int(self.count * 0.2),
        )

        lognorm_widths = lognorm.rvs(
            0.9311,
            loc=0,
            scale=0.0011,
            size=int(self.count * 0.8),
        )
        lognorm_widths[lognorm_widths < 0.0001] = 0.0001
        width = np.concatenate((width, lognorm_widths))
        # -------------------------------------------------------------------------------

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

        spectral_running = np.linspace(
            self.component_0["spectral_running"]["sr_lower_value"],
            self.component_0["spectral_running"]["sr_upper_value"],
            self.count,
        )

        central_freq = np.linspace(
            self.input_params["central_freq"]["lower_value"],
            self.input_params["central_freq"]["upper_value"],
            self.count,
        )

        bandwidth = np.linspace(
            self.input_params["bandwidth"]["lower_value"],
            self.input_params["bandwidth"]["upper_value"],
            self.count,
        )

        # Shuffle parameters
        random.shuffle(DM)
        random.shuffle(SM)
        random.shuffle(width)
        random.shuffle(fluence)
        random.shuffle(spectral_index)
        random.shuffle(spectral_running)
        random.shuffle(central_freq)
        random.shuffle(bandwidth)

        data = []
        frb_metadata = []
        min_max = []

        for i in range(self.count):

            frb_dict = {
                "id": f"frb_{i}",
                "dm": DM[i],
                "sm": SM[i],
                "components": [
                    {
                        "arrival_time": arrival_time,
                        "width": width[i],
                        "fluence": fluence[i],
                        "spectral_index": -spectral_index[i],
                        "spectral_running": spectral_running[i],
                    }
                ],
            }

            img = self.generate_frb(
                frb_dict,
                central_freq[i],
                bandwidth[i],
                self.snr,
            )

            data.append(img)
            frb_metadata.append(frb_dict)
            min_max.append((img.max(), img.min()))

        return data, frb_metadata, min_max

    # -------------------------------------------------

    def generate_frb(
        self,
        frb_dict,
        central_frequency,
        bandwidth_used,
        snr,
    ):
        """
        Generate a single Type-III FRB dynamic spectrum.
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
                comp["spectral_index"],
                comp["arrival_time"],
            )

            comp_ts = np.zeros_like(ts)
            frb_pulse.add_to_timestream(comp_ts, 2.8, 3.3)

            obs_freq = [FREQ_LO_MHZ, FREQ_HI_MHZ]
            comp_ts = gaussian_filter(
                comp_ts,
                comp_ts.shape,
                central_frequency,
                bandwidth_used,
                obs_freq,
            )

            ts += comp_ts

        ts /= 100.0

        noise_rms = calculate_noise_rms(ts, snr)

        if self.noise:
            ts += np.random.normal(0, noise_rms, ts.shape)

        ts = np.flip(ts, axis=0)

        s = CROP_CENTER - CROP_HALF_WIDTH
        e = CROP_CENTER + CROP_HALF_WIDTH

        return ts[:, s:e]


####################################################################################################

if __name__ == "__main__":

    start = time.time()

    simulator = CreateFRB()
    data, frb_header, min_max = simulator.simulate_frbs(snr=50.0)

    write_out_frbs(data, frb_header, min_max, "C")
    
    print(f"Simulation completed in {time.time() - start:.1f} s")

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
        plt.savefig(f"type_C_images/frb_type_C_{i}.png", dpi=150)
        plt.clf()

    
                
                
                
                
                

