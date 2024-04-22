#include "dsp.h"
#include "tensor.h"

int main() {
    Tensor<float> filter_banks = Tensor<float>::fromFile("filter_banks/filter_banks.npy").squeeze(0).transpose().contiguous();
    Tensor<float> signal = Tensor<float>::arange(0, 2559);
    int stride = 160;
    int n_fft = 512;
    int padding = n_fft / 2;
    Tensor<float> window_factor = window(Tensor<float>::full({n_fft}, 1.0), "hann");

    signal = signal.pad({padding, padding}, 0);
    signal = signal.frame(n_fft, stride, 0);
    signal = signal * window_factor;

    Tensor<std::complex<double>> complex_signal = signal.template astype<std::complex<double>>();

    Tensor<std::complex<double>> spectrogram;
    for (size_t i = 0; i < 16; i++) {
        Tensor<std::complex<double>> frame = complex_signal[Slice(i, i + 1)];
        frame = frame.squeeze(0);
        Tensor<std::complex<double>> fft_result = fft(frame);
        if (i == 0) {
            spectrogram = fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0);
        } else {
            spectrogram = spectrogram.concat(fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0), 0);
        }
    }

    spectrogram = spectrogram.abs().pow(2);
    Tensor<float> float_spectrogram = spectrogram.template astype<float>();
    Tensor<float> mel_spectrogram = float_spectrogram.matmul(filter_banks);
    mel_spectrogram = (mel_spectrogram + 1e-6).log();

    mel_spectrogram.printDims();
    mel_spectrogram.printData();

    return 0;
}