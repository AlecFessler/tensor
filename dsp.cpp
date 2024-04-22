#include "dsp.h"
#include "tensor.h"

int main() {
    Tensor<float> filter_banks = Tensor<float>::fromFile("filter_banks/filter_banks.npy");
    Tensor<float> signal = Tensor<float>::arange(0, 2559);
    int stride = 160;
    int n_fft = 512;
    int padding = n_fft / 2;
    Tensor<float> window_factor = window(Tensor<float>::full({n_fft}, 1.0), "hann");

    signal = signal.pad({padding, padding}, 0);
    signal = signal.frame(n_fft, stride, 0);
    signal = signal * window_factor;

    Tensor<std::complex<double>> signal_complex = signal.template astype<std::complex<double>>();

    Tensor<std::complex<double>> concated;
    // Loop 16 times, slicing the signal into 16 frames, and computing the FFT of each frame and concatenating the results
    for (size_t i = 0; i < 16; i++) {
        Tensor<std::complex<double>> frame = signal_complex[Slice(i, i + 1)];
        frame.printDims();
        Tensor<std::complex<double>> fft_result = fft(frame);
        fft_result.printDims();
        if (i == 0) {
            concated = fft_result;
        } else {
            concated = concated.concat(fft_result, 0);
        }
        concated.printDims();
    }

    return 0;
}