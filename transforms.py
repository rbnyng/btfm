import numpy as np
import torch
import logging
# contains the transforms for sampling the data
# each sample is (time, bands) and (time) for the mask
# the transform takes the bands and masks out invalid samples
# then it takes two samples (by time) and returns them

class SampleValidPixels():
    def __init__(self, sample_size, seed=None):
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)

    # mask_sample is (time), band_sample is (time, bands) and doys is the
    # day of year for each time
    def __call__(self, mask_sample, band_sample, doys):
        result_samples = torch.zeros((2, self.sample_size, band_sample.shape[1]))
        doy_samples = torch.zeros((2, self.sample_size), dtype=torch.long)
        doys = doys.long()

        valid_pixel_idx = np.nonzero(mask_sample).flatten()

        s0_random_idx = self.rng.choice(valid_pixel_idx, self.sample_size, replace=False)
        s1_random_idx = self.rng.choice(valid_pixel_idx, self.sample_size, replace=False)

        # sort the indexes so they are in ascending order
        s0_random_idx.sort()
        s1_random_idx.sort()

        result_samples[0] = band_sample[s0_random_idx]
        result_samples[1] = band_sample[s1_random_idx]

        doy_samples[0] = doys[s0_random_idx] // 7
        doy_samples[1] = doys[s1_random_idx] // 7

        return result_samples, doy_samples

class DummyTransform():
    def __init__(self, sample_size, seed=None):
        self.sample_size = sample_size

    def __call__(self, mask_sample, band_sample):
        result_samples = torch.zeros((2, self.sample_size, band_sample.shape[1]))

        valid_pixel_idx = np.nonzero(mask_sample).flatten()

        result_samples[0] = band_sample[valid_pixel_idx][:self.sample_size]
        result_samples[1] = band_sample[valid_pixel_idx][self.sample_size:2*self.sample_size]

        return result_samples