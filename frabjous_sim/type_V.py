##Script to simulate type V bursts

from simpulse import single_pulse as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import yaml
import random
import time
from scipy.stats import lognorm

from utils import (
    calculate_noise_rms,
    gaussian_filter,
    write_out_frbs,
)

####global parameters for generating dynamic spectra 
PULSE_NT = 512
NFREQ = 256
FREQ_LO_MHZ = 400.0
FREQ_HI_MHZ = 800.0


class create_frb:
    """
    Class to simulate Type-C2 FRBs with multiple components showing
    downward drifting (sad trombone) patterns.
    """
    
    #### Read configuration file
    with open("config_type_V.yml") as file:
        try:
            input_params = yaml.safe_load(file)
            print(input_params)
        except yaml.YAMLError as exc:
            raise RuntimeError("Error reading YAML config") from exc

    noise = input_params["add_noise"]
    snr = input_params["snr"]

    dm_range = input_params["dm_range"]
    scattering_measure = input_params["scattering_measure"]
    component_0 = input_params["component_0"]

    count = input_params["count"]
    dm_lo = dm_range["lower_dm_value"]
    dm_up = dm_range["upper_dm_value"]

    drift_rates = input_params["drift_rates"]
    drift_rate_lo = drift_rates["lo"]
    drift_rate_up = drift_rates["up"]

    central_freq = input_params["cent_freq"]
    bandwidth = input_params["bandwidth"]
    arr_time_sep = input_params["arr_time_sep"]

    # ----------------------------------------------------------

    def simulate_frbs(self):
        """
        Generate a set of simulated Type-C2 FRBs.

        Returns
        -------
        simulated_FRBs : list of 2D numpy arrays
        frb_information : list of dictionaries
        min_max : list of (max, min) tuples
        """

        DM = np.linspace(self.dm_lo, self.dm_up, self.count)
        SM = np.linspace(
            self.scattering_measure["sm_lower"],
            self.scattering_measure["sm_upper"],
            self.count,
        )

        arrival_time = self.component_0["arrival_time"]

        # width distribution obatined from the first chime FRB catalog
        lognorm_widths = lognorm.rvs(
            0.9311,
            loc=0,
            scale=0.0011,
            size=self.count,
            random_state=40,
        )
        lognorm_widths[lognorm_widths < 0.0001] = 0.0001
        lognorm_widths[lognorm_widths > 0.01] = np.linspace(
            0.001, 0.01, len(lognorm_widths[lognorm_widths > 0.01])
        )
        width = lognorm_widths

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

        num_of_comp = [2, 3, 4, 5]
        fluence_ratio = np.linspace(0.5, 1.0, self.count)

        drift_rate = np.linspace(
            self.drift_rate_lo,
            self.drift_rate_up,
            self.count,
        )

        arrival_time_sep = np.linspace(
            self.arr_time_sep["lo"],
            self.arr_time_sep["hi"],
            self.count,
        )

        # Shuffle parameters
        for arr in [
            DM, SM, width, fluence,
            arrival_time_sep, fluence_ratio, drift_rate
        ]:
            random.shuffle(arr)

        simulated_FRBs = []
        frb_information = []
        min_max = []

        for i in range(self.count):
            frb_dict = {
                "dm": DM[i],
                "sm": SM[i],
                "components": [
                    {
                        "arrival_time": arrival_time,
                        "width": width[i],
                        "fluence": fluence[i],
                        "spectral_index": spectral_index[i],
                        "spectral_running": spectral_running[i],
                    },
                ],
            }

            # Add additional components
            for k in range(1, 5):
                frb_dict["components"].append(
                    {
                        "arrival_time": arrival_time + np.sum(arrival_time_sep[i-k:i]),
                        "width": self.choosing_width(width, arrival_time_sep[i-k]),
                        "fluence": fluence[i] * np.random.choice(fluence_ratio),
                        "spectral_index": spectral_index[i],
                        "spectral_running": spectral_running[i],
                    }
                )

            frb_dict["id"] = f"frb_{i}"
            frb_information.append(frb_dict)

            FRB = self.generate_frb(
                frb_dict,
                random.choice(num_of_comp),
                self.noise,
                random.choice(drift_rate),
            )

            simulated_FRBs.append(FRB)
            min_max.append((np.amax(FRB), np.amin(FRB)))

        return simulated_FRBs, frb_information, min_max

    def generate_frb(self, frb_dict, comp, noise, drift_rate):
        """
        Generate a single Type-C2 FRB dynamic spectrum.
        """

        waterfall = np.zeros((NFREQ, PULSE_NT))

        central_frequencies = np.linspace(
            self.central_freq["lo"],
            self.central_freq["up"],
            self.count,
        )
        bandwidth_values = np.linspace(
            self.bandwidth["lo"],
            self.bandwidth["up"],
            self.count,
        )

        central_freq_bri = random.choice(central_frequencies)

        for i in range(comp):
            frb_pulse = sp(
                PULSE_NT,
                NFREQ,
                FREQ_LO_MHZ,
                FREQ_HI_MHZ,
                frb_dict["dm"],
                frb_dict["sm"],
                frb_dict["components"][i]["width"],
                frb_dict["components"][i]["fluence"],
                0.0,
                frb_dict["components"][i]["arrival_time"],
            )

            central_freq = (
                central_freq_bri
                + drift_rate * 1000
                * (
                    frb_dict["components"][i]["arrival_time"]
                    - frb_dict["components"][0]["arrival_time"]
                )
            )

            ts = np.zeros((frb_pulse.nfreq, PULSE_NT))
            frb_pulse.add_to_timestream(ts, 2.8, 3.3)

            ts = gaussian_filter(
                ts,
                ts.shape,
                central_freq,
                random.choice(bandwidth_values),
                [FREQ_LO_MHZ, FREQ_HI_MHZ],
            )

            waterfall += ts

        waterfall /= 100.0
        noise_rms = calculate_noise_rms(waterfall, self.snr)

        if noise:
            waterfall += np.random.normal(0, noise_rms, waterfall.shape)

        waterfall = np.flip(waterfall, axis=0)

        index = 230
        return waterfall[:, index - 128 : index + 128]

    def choosing_width(self, arr, value):
        valid = [x for x in arr if x < (value / 2)]
        return random.choice(valid)


if __name__ == "__main__":

    start = time.time()

    simulator = create_frb()
    data, frb_header, min_max = simulator.simulate_frbs()

    write_out_frbs(data, frb_header, min_max, "C2")
    
    print(f"Simulation completed in {time.time() - start:.1f} s")

## comment out this part if you want to save and check the generated samples in png format

#    for i, img in enumerate(data[:]):
#        plt.imshow(img, aspect="auto")
#        plt.colorbar()
#        plt.savefig(f"type_C2_images/frb_type_C2_{i}.png", dpi=150)
#        plt.clf()

