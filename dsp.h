#ifndef DSP_H
#define DSP_H

#include <complex>

#include "tensor.h"

template <typename T>
Tensor<T> window(const Tensor<T>& signal, const std::string& window_type = "hann") {
    /**
     * Apply a window function to the input signal
     * 
     * Currently only supports the Hann window function
     * 
     * @param signal: Input signal
     * @param window_type: Type of window function (e.g., "hann", "hamming")
     * @return: Windowed signal
    */
    if (window_type == "hann") {
        auto size = signal.size();
        Tensor<T> hann = Tensor<T>::arange(0, size);
        hann = 0.5 * (1 - ((hann * (2 * M_PI / size)).cos()));
        return signal * hann;
    }
    return signal;
}

template <typename T>
Tensor<T> fft(Tensor<T>& signal) {
    /**
     * Computes the Fast Fourier Transform (FFT) of the input signal
     * 
     * @param signal: Input signal
     * @return: FFT of the input signal
     * 
     * @throws std::invalid_argument: If the size of the input signal is not a power of 2
    */
    int size = signal.size();
    if (size == 1) {
        return signal;
    } else if (size & (size - 1)) {
        throw std::invalid_argument("FFT size must be a power of 2");
    }

    Tensor<std::complex<double>> result({size});
    std::complex<double> omega = std::exp(std::complex<double>(0, -2 * M_PI / size));
    std::complex<double> omega_n = 1;

    Tensor<std::complex<double>> even = signal[Slice(0, size, 2)];
    Tensor<std::complex<double>> odd = signal[Slice(1, size, 2)];
    even = fft(even);
    odd = fft(odd);

    for (int i = 0; i < size / 2; i++) {
        result[{i}] = even[{i}] + omega_n * odd[{i}];
        result[{i + size / 2}] = even[{i}] - omega_n * odd[{i}];
        omega_n *= omega;
    }

    return result;
}

template <typename T>
Tensor<T> asrPreprocessor(Tensor<T> signal, Tensor<T> filter_banks, int stride=160, int n_fft=512, int padding=256) {

    Tensor<T> first_elem = signal[Slice(0, 1)];
    signal = first_elem.concat(signal[Slice(1)] - 0.97 * signal[Slice(0, signal.size() - 1)], 0);
    signal = signal.pad({padding, padding}, 0);
    signal = signal.frame(n_fft, stride, 0).contiguous(); // make contiguous after framing

    Tensor<double> window_factor = window(Tensor<double>::full({n_fft}, 1.0), "hann");
    Tensor<double> windowed_signal = signal.astype<double>() * window_factor; // cast to double before windowing

    Tensor<std::complex<double>> complex_signal = windowed_signal.template astype<std::complex<double>>();
    Tensor<std::complex<double>> spectrogram;
    Tensor<std::complex<double>> frame;
    Tensor<std::complex<double>> fft_result;

    for (int i = 0; i < complex_signal.size(); i++) {
        frame = complex_signal[Slice(i, i + 1)].squeeze(0);
        fft_result = fft(frame);
        if (i == 0) {
            spectrogram = fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0);
        } else {
            spectrogram = spectrogram.concat(fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0), 0);
        }
    }

    Tensor<T> power_spectrum = spectrogram.abs().pow(2).astype<T>(); // casted to T
    Tensor<T> mel_spectrogram = power_spectrum.matmul(filter_banks);
    mel_spectrogram = (mel_spectrogram + 1e-6).log().transpose().contiguous().unsqueeze(0);

    return mel_spectrogram;
}

#endif // DSP_H