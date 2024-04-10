import numpy as np
import matplotlib.pyplot as plt


def visualize_frames(file_path):
    """
    Visualizes frames from the given files in subplots.
    Args: file_paths: Paths to the files containing the frame data.
    """
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    frames = np.load(file_path, allow_pickle=True)

    for i, frame in enumerate(frames):
        ax = axes[i]
        ax.plot(frame)
        ax.set_title(f"Frame {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_xlim(0, len(frame))
        ax.set_ylim(np.min(frame), np.max(frame))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_frames("data/frames/validation/S16/1_7.npy")
