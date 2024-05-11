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

#endif // DSP_H