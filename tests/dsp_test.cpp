#include "../dsp.h"
#include "../tensor.h"

template <typename T>
T mse(Tensor<T> x, Tensor<T> y) {
    return (x - y).pow(2).mean();
}

int main() {
    Tensor<float> filter_banks = Tensor<float>::fromFile("../filter_banks/filter_banks.npy");
    
    // compare arange with numpy arange
    Tensor<float> signal = Tensor<float>::arange(0, 2559);
    Tensor<float> numpy_signal = Tensor<float>::fromFile("../tests/signal.npy");
    float mse_result = mse(signal, numpy_signal);
    std::cout << "MSE between arange and numpy arange: " << mse_result << std::endl;

    int stride = 160;
    int n_fft = 512;
    int padding = n_fft / 2;

    // compare preemp with numpy preemp
    Tensor<float> first_elem = signal[Slice(0, 1)];
    signal = first_elem.concat(signal[Slice(1)] - 0.97 * signal[Slice(0, signal.size() - 1)], 0);
    Tensor<float> numpy_preemp = Tensor<float>::fromFile("../tests/preemp.npy");
    mse_result = mse(signal, numpy_preemp);
    std::cout << "MSE between preemp and numpy preemp: " << mse_result << std::endl;

    // compare pad with numpy pad
    signal = signal.pad({padding, padding}, 0);
    Tensor<float> numpy_pad = Tensor<float>::fromFile("../tests/padded.npy");
    mse_result = mse(signal, numpy_pad);
    std::cout << "MSE between pad and numpy pad: " << mse_result << std::endl;

    // compare frame with numpy frame
    signal = signal.frame(n_fft, stride, 0).contiguous(); // ensure contiguous here
    Tensor<float> numpy_frame = Tensor<float>::fromFile("../tests/framed.npy");
    mse_result = mse(signal, numpy_frame);
    std::cout << "MSE between frame and numpy frame: " << mse_result << std::endl;

    // compare window with numpy window
    Tensor<double> window_factor = window(Tensor<double>::full({n_fft}, 1.0), "hann");
    Tensor<double> windowed_signal = signal.astype<double>() * window_factor; // ensure types match here
    Tensor<double> numpy_window = Tensor<double>::fromFile("../tests/windowed.npy");
    double mse_result_d = mse(windowed_signal, numpy_window);
    std::cout << "MSE between window and numpy window: " << mse_result << std::endl;

    // compare fft with numpy fft
    Tensor<std::complex<double>> complex_signal = windowed_signal.template astype<std::complex<double>>();
    Tensor<std::complex<double>> spectrogram;
    Tensor<std::complex<double>> frame;
    Tensor<std::complex<double>> fft_result;

    for (int i = 0; i < 16; i++) {
        frame = complex_signal[Slice(i, i + 1)].squeeze(0);
        fft_result = fft(frame);
        if (i == 0) {
            spectrogram = fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0);
        } else {
            spectrogram = spectrogram.concat(fft_result[Slice(0, n_fft / 2 + 1)].unsqueeze(0), 0);
        }
    }

    Tensor<std::complex<double>> numpy_fft = Tensor<std::complex<double>>::fromFile("../tests/spectrogram.npy");
    std::complex<double> mse_result_c = mse(spectrogram, numpy_fft);
    std::cout << "MSE between fft and numpy fft: " << mse_result << std::endl;

    // compare power spectrum with numpy power spectrum
    Tensor<double> power_spectrum = spectrogram.abs().pow(2).astype<double>();
    Tensor<double> numpy_power_spectrum = Tensor<double>::fromFile("../tests/powers.npy");
    mse_result_d = mse(power_spectrum, numpy_power_spectrum);
    std::cout << "MSE between power spectrum and numpy power spectrum: " << mse_result << std::endl;

    // compare typecast to float with numpy float power spectrum
    Tensor<float> float_spectrogram = power_spectrum.astype<float>();
    Tensor<float> np_float_spectrogram = Tensor<float>::fromFile("../tests/casted_powers.npy");
    mse_result = mse(float_spectrogram, np_float_spectrogram);
    std::cout << "MSE between float spectrogram and numpy float spectrogram: " << mse_result << std::endl;

    // compare mel spectrogram with numpy mel spectrogram
    Tensor<float> mel_spectrogram = float_spectrogram.matmul(filter_banks);
    Tensor<float> numpy_mel_spectrogram = Tensor<float>::fromFile("../tests/mel_spectrogram.npy");
    mse_result = mse(mel_spectrogram, numpy_mel_spectrogram);
    std::cout << "MSE between mel spectrogram and numpy mel spectrogram: " << mse_result << std::endl;

    // compare mel spectrogram no zeroes with numpy mel spectrogram no zeroes
    mel_spectrogram = mel_spectrogram + 1e-6;
    numpy_mel_spectrogram = Tensor<float>::fromFile("../tests/mel_spectrogram_no_zero.npy");
    mse_result = mse(mel_spectrogram, numpy_mel_spectrogram);
    std::cout << "MSE between mel spectrogram no zeroes and numpy mel spectrogram no zeroes: " << mse_result << std::endl;

    // compare log mel spectrogram with numpy log mel spectrogram
    mel_spectrogram = mel_spectrogram.log();
    numpy_mel_spectrogram = Tensor<float>::fromFile("../tests/log_mel_spectrogram.npy");
    mse_result = mse(mel_spectrogram, numpy_mel_spectrogram);
    std::cout << "MSE between log mel spectrogram and numpy log mel spectrogram: " << mse_result << std::endl;

    // compare final processed signal with numpy final processed signal
    Tensor<float> final_signal = mel_spectrogram.transpose().contiguous().unsqueeze(0);
    Tensor<float> numpy_final_signal = Tensor<float>::fromFile("../tests/processed_signal.npy");
    mse_result = mse(final_signal, numpy_final_signal);
    std::cout << "MSE between final processed signal and numpy final processed signal: " << mse_result << std::endl;

    return 0;
}