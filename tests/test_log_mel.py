from voicebox_pytorch import load_audio


def main():
    waveform, sr = load_audio("./sample-1.wav")


if __name__ == "__main__":
    main()
