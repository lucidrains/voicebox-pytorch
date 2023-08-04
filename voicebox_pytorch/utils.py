import torchaudio


def load_audio(path):
    waveform, original_sr = torchaudio.load(path)
    print(waveform.shape, original_sr)
    return waveform
