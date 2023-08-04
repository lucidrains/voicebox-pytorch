from voicebox_pytorch.utils import load_audio, log_mel
import librosa
import matplotlib.pyplot as plt


def plot_log_mel(log_mel_spec, sr):
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(
        log_mel_spec, sr=sr, hop_length=160, win_length=640,
        fmin=0, fmax=8000, x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Log Mel Spectrogram')
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig("sample-1.png")


def main():
    waveform, sr = load_audio("./sample-1.wav")
    mel_frames = log_mel(waveform, sr)
    plot_log_mel(mel_frames.numpy(), sr)
    breakpoint()


if __name__ == "__main__":
    main()
