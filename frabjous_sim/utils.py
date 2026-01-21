from simpulse import single_pulse as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use("Agg")
import matplotlib.image as mpimg
import sys
import json 

 


def gaussian_filter( array, shape, central_freq, bandwidth, obs_freq ):
        """
        This function takes 2-D numpy array and multiplies a gaussian fuction to provide
        different narrow band emission feature in the burst.
        """
        ## create a gaussain function
        
        number_of_channels = shape[0]         # number of samples along the axis
        lowest_freq = obs_freq[0]
        highest_freq = obs_freq[1]
        obs_bandwidth = highest_freq - lowest_freq
        freq_res =  obs_bandwidth / number_of_channels
        normalise= sum(sum(array))
        for i in range(0,number_of_channels): 
            array[i] = array[i] * gaussian( i*freq_res + lowest_freq ,central_freq , bandwidth )
        return array *( normalise / sum( sum(array) ) )   


def gaussian(freq,central_freq , bandwidth):
    """
    This function returns a gaussian  for a given central point 
    and FWHM 
    """
    x = freq - central_freq
    sigma = bandwidth / 2     
    return np.exp( - x**2 / (2*sigma**2) )       

def calculate_noise_rms(img , SNR) : 
    '''
    Calcuates RMS noise for the waterfall plot for the FRB burst 
    we use the 90% of the pulse emission that where we take cumulative sum of the time series along the time axsi then 
    we take 5% to 95% of the maximum cumulative flux density to calculate the width 
    '''
    pulse =[ sum(img[:,i]) for i in range( 0, np.shape(img) [1] ) ]

    time_series = np.cumsum(pulse)
    effective_width = np.where(time_series > 0.95*np.max(time_series) )[0][1] - np.where(time_series> 0.05*np.max(time_series) )[0][1] 
    width_factor = np.sqrt(effective_width ) 	
    return sum(sum(img)) / (np.sqrt(np.shape(img)[0])*SNR*width_factor)

def apply_spectral_index_and_running(array, shape, spectral_index, spectral_running):
    """
    will apply spectral properties to the broad band pulse
    Input --

    Takes the 2-D numpy array with axes representing frequency and time.
    This array contains contains the pulse with broad band properties.

    Output
    2-D Numpy array with specific spectral properties for each component

    """
    print( sum(sum(array)) )
    int_flux = sum( sum(array) )
    non_zero = np.where(array[0])
    N = shape[0]  # number of freq channels
    

    for i in range(0, N):
        freq = 400 + (i * 400 / N)
        array[i] = array[i] * (freq / 400) ** (
            spectral_index + (spectral_running * np.log( freq / 400 ) )
        )
    renormalise_factor = int_flux / (sum(sum(array)))
    array = array * renormalise_factor
    return array

def write_out_frbs(data,frb_header,min_max,typ) : 
    '''
    writing out onformation for FRBs into numpy array , its parameters in a header to a  json file
    and maxima minima of the numpy to a txt file
    '''
    f = open("simdata/min_max_type_"+ typ +".txt", "w" )
    for max, min in min_max:
        f.write(str(max) + " , " + str(min) + "\n")
    f.close() 
    
    # numpy arrays for each FRBs saved as compressed numpy file in npz format 
    np.savez_compressed("simdata/simulatefrbs_type_"+ typ +".npz", data)
    
    #writing out the disctionary for each FRB
    with open("simdata/frb_header_type_"+ typ +".json", "w") as final:
        json.dump(frb_header, final)

def fraunhoffer_pattern(image,theta):
    '''
    This function implements fraunhoffer diffraction pattern for a particular theta
    at a given frequency range 
    
    '''
    ##define the speed of light 
    c = 3 * 10**8
    freq  = np.linspace(400,800,256)    #frequencies in MHz
    aperture = 80   ## CHIME largest baseline
    int_flux = sum(sum(image))
    print(int_flux)
    for i in range(0 ,len(image)):
        fact = freq[i]                    # factor is the ratio of frequency (in MHz) to speed of light c (represents the wavelength in the diffraction pattern )
        #print(fact)
        phi =  np.pi * aperture * (fact/300) *np.sin( np.deg2rad(theta) )
        image[i] = image[i]* (np.sin(phi) / phi )**2
    
    print(sum(sum(image)))
    image = image *( int_flux/sum(sum(image)) )
    print( sum(sum(image))  )
    return image     
    
