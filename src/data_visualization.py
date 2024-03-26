import numpy as np
import matplotlib.pyplot as plt


def visualize_frames(file_paths):
    """
    Visualizes frames from the given files in subplots.
    Args: file_paths: Paths to the files containing the frame data.
    """
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    global_min = float("inf")
    global_max = float("-inf")

    for i, file_path in enumerate(file_paths):
        frame = np.load(file_path)

        ax = axes[i // 2, i % 2]
        ax.plot(frame)
        ax.set_title(f"Frame {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)

        local_min = frame.min()
        local_max = frame.max()
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max

    for ax in axes.flat:
        ax.set_xlim(0, len(frame))
        ax.set_ylim(global_min - 0.1, global_max + 0.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_paths = [
        "data/frames/S17/BVP/1_BVP_0.npy",
        "data/frames/S17/BVP/1_BVP_1.npy",
        "data/frames/S17/BVP/1_BVP_2.npy",
        "data/frames/S17/BVP/1_BVP_3.npy",
    ]

    visualize_frames(file_paths)
