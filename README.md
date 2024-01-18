# Noise-CT-Scans

## Hounsfield scale  
The pixel values of this CT scan are expressed in Hounsfield Units:
$$HU(x,y) = 1000 \cdot \frac{\mu{(x,y)} - \mu_{water}}{\mu_{water}-\mu_{air}}$$
where $\mu$ is the attenuation coefficient of the material. The linear attenuation coefficient is defined based on how the intensity of a photon beam decays as it passes a distance $x$ through a material $I=I_{0}e^{-\mu x}$. Note that $\mu$ depends on the energy of the photon beam, and in a CT scan photons usually have energies 100 keV.
## Noise assesment 
Basic concept of signal and noise:    
![](https://github.com/MKastek/Noise-CT-Scans/blob/add0e48e8d3c5f1b0bde1b278ce0def490527220/images/noise-assesment.PNG)  
The noise distributions within the object and in the background are characterized by normal distributions defined by their standard deviations $\sigma$, with the shift between the two distributions corresponding to the signal amplitude: $|s_{1}-s_{2}|$. Although the signal-to-noise ratio is an important parameter in determining the detectability of an object, the SNR does not completely characterize noise. These two images with exactly the same noise level as measured by the standard
deviation $\sigma$ however, these two images have dramatically different appearances to the observer.  
![](https://github.com/MKastek/Noise-CT-Scans/blob/add0e48e8d3c5f1b0bde1b278ce0def490527220/images/noise-assesment-sigma.PNG)  
The differences between images are due to the noise texture, that is, the spatial-frequency distribution of the noise is different in these two images.  
## Noise Power Spectrum - NPS 
The noise-power spectrum (NPS) is a useful measure that provides a more complete description of noise than the simple standard deviation. It describes the noise variance as a function of spatial frequency and therefore characterizes noise texture.  
## NPS 2D   
![](https://github.com/MKastek/Noise-CT-Scans/blob/9bb4ce518d1db5e4656f3e1a2478faca685bf3ee/images/roi.PNG)  
NPS can be calculated with the formula:
$$NPS(f_{x},f_{y})=\frac{1}{N}\frac{\Delta_{x}\Delta_{y}}{N_{x}N_{y}}\sum_{i=1}^{N}|DFT_{2D}[I_{i}(x,y)-\bar{I_{i}}]|^{2}$$
where:  
$N$ - the number of ROIs  
$I_{i}(x,y)$ - the signal in the $i^{th}$ ROI  
$\bar{I}$ - mean of $I_{i}(x,y)$  
$\Delta_{x}, \Delta_{y}$ - pixel size  
Result of NPS in spatial frequency $f_{x}, f_{y}$  
![](https://github.com/MKastek/Noise-CT-Scans/blob/7c03f51d6ad282a6db71883360067f4a345c1877/images/NPS-2D.PNG)  
## NPS 1D  
The $f_{x}$ and $f_{y}$ frequencies in the 2D NPS can be collapsed to a 1D radial frequency, $f_{r}$ by radially averaging using:  
$$f_{r} = \sqrt{f_{x}^{2}+f_{y}^2}$$  
![](https://github.com/MKastek/Noise-CT-Scans/blob/9bb4ce518d1db5e4656f3e1a2478faca685bf3ee/images/NPS-1D.PNG)  
The initial positive slope of this curve results from the ramp filtering that is used in filtered-back-projection reconstruction, and the negative slope at higher spatial frequencies occurs due to the roll-off properties of the reconstruction kernel used to dampen high-frequency noise in the images.
## Denoising 
## Denosing with Deep Image Prior
## Configuration
## Run training
## Run evaluation
