import numpy as np
import matplotlib.pyplot as plt


def visualize_frames(file_paths):
    """
    Visualizes frames from the given files in subplots.
    Args: file_paths: Paths to the files containing the frame data.
    """
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    for i, file_path in enumerate(file_paths):
        frame = np.load(file_path)

        ax = axes[i // 2, i % 2]
        ax.plot(frame)
        ax.set_title(f"Frame {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_xlim(0, len(frame))
        ax.set_ylim(frame.min(), frame.max())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    paths = [
        "data/frames/S5/EDA/0_EDA_0.npy",
        "data/frames/S5/EDA/0_EDA_2319.npy",
        "data/preprocessed_frames/S5/EDA/0_EDA_0_preprocessed.npy",
        "data/preprocessed_frames/S5/EDA/0_EDA_2319_preprocessed.npy",
    ]

    visualize_frames(paths)
