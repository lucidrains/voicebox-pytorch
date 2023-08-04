
import torchaudio
import torch


def load_audio(path, sr=16_000):
    waveform, original_sr = torchaudio.load(path)
    print("Original:", waveform.shape, original_sr)

    # Resampling
    if original_sr != sr:
        resampler = torchaudio.transforms.Resample(original_sr, sr)
        waveform = resampler(waveform)
        print("After resampling:", waveform.shape, sr)

    # Making mono
    if waveform.shape[0] > 1:  # If audio has more than one channel
        waveform = torch.mean(waveform, dim=0)
        print("After making mono:", waveform.shape)

    print("Waveform is {} seconds long".format(len(waveform) / sr))
    return waveform, sr


def log_mel(audio, sr, n_fft=1024):
    # 1024-point STFT with a 640-sample (40ms) analysis window and 160-sample (10ms) shift
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=640, hop_length=160, window_fn=torch.hann_window
    )
    spectrogram = stft_transform(audio)

    # Create an 80 dimension Mel filter with a cut-off frequency of 8kHz
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=80, sample_rate=sr, n_stft=n_fft // 2+1, f_max=8000
    )
    mel_spectrogram = mel_transform(spectrogram)

    # Apply log to get log Mel spectrogram
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    return log_mel_spectrogram
