#ifndef cppDsp_h
#define cppDsp_h

#include "complexNum.h"

// Apply preemphasis to the input audio signal:
// Preemphasis coefficient: 0.97
// Operation: signal[1:] - 0.97 * signal[:-1]
// Segment the audio signal into overlapping frames:
// Frame length: 400 samples (25 ms)
// Frame stride: 160 samples (10 ms)
// Padding: Reflect mode
// Apply a Hann window to each frame:
// Window length: 400 samples
// Compute the FFT of each windowed frame:
// FFT size: 512 points
// Omit the redundant second half of the FFT output (onesided=True)
// Compute the power spectrum of each FFT frame:
// Operation: Square the magnitude of each FFT coefficient
// Apply a mel filterbank to the power spectra:
// Number of mel bins: 80
// Frequency range: 0 Hz to Nyquist frequency (sample_rate / 2)
// Mel scale: Slaney
// Normalization: Slaney (area normalized)
// Apply log scaling to the mel spectrogram:
// Operation: log(mel_spectrogram + 1e-6)
// (Optional) Pad the mel spectrogram to a desired length:
// Pad to a multiple of a specified value (e.g., pad_to=16)
// Padding value: 0
// These steps should produce a log-scaled mel spectrogram with dimensions (num_mel_bins, num_frames).

// Additional parameters:

// Sample rate: 16000 Hz
// Number of mel bins: 80
// FFT size: 512 points
// Frame length: 400 samples (25 ms)
// Frame stride: 160 samples (10 ms)
// Preemphasis coefficient: 0.97
// Log scaling constant: 1e-6

// Tensor type and operations
// - Define a tensor class or use a library like Eigen or Armadillo for efficient tensor operations
// - Implement basic tensor operations like addition, multiplication, and slicing

// FFT
// - Implement a Fast Fourier Transform (FFT) algorithm or use a library like FFTW or KissFFT
// - Provide functions for forward and inverse FFT

// Windowing function
// - Implement a function to apply a window (e.g., Hann, Hamming) to each frame
// - Provide common window types or allow custom window functions

// Function to load mel filterbanks from CSV file
// - Implement a function to load pre-computed mel filterbanks from a CSV file
// - Handle different file formats and ensure compatibility with your tensor type

// Pre-emphasis function
// - Implement a function to apply pre-emphasis to the input signal
// - Allow specifying the pre-emphasis coefficient

// Log scale function
// - Implement a function to apply logarithmic scaling to the mel spectrogram
// - Handle small or negative values using a suitable epsilon value

// Mel spectrogram computation
// - Implement a function to compute the mel spectrogram from the input audio signal
// - Combine the above functions (padding, segmentation, windowing, FFT, power spectrogram, mel scaling, log scaling)
// - Allow specifying the necessary parameters like sample rate, frame length, hop length, mel filterbanks, etc.

// Utility functions
// - Implement utility functions for reading audio files, converting between data types, etc.
// - Provide functions for common audio preprocessing tasks like resampling, normalization, or silence removal


#endif // cppDsp_h