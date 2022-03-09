from abcd.denoiser import Denoiser

x = Denoiser(directory='images', output='images/denoised/')
x.denoise()