import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def animate_video(video):

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    def animate(frame):
        ax.clear()
        ax.imshow(video[frame], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Frame {frame}')

    ani = animation.FuncAnimation(fig, animate, frames=range(len(video)), interval=5, repeat=False)

    return ani

def interpolate_linear(observed_times, new_times, states):
    new_states = np.zeros((len(new_times), *states.shape[1:]))

    for j, time in enumerate(new_times):
        # find the two closest observed times
        idx = np.searchsorted(observed_times, time)
        if idx == 0:
            new_states[j] = states[0]
        elif idx == len(observed_times):
            new_states[j] = states[-1]
        else:
            # linearly interpolate between the two closest observed times
            alpha = (time - observed_times[idx-1]) / (observed_times[idx] - observed_times[idx-1])
            new_states[j] = alpha * states[idx] + (1 - alpha) * states[idx-1]

    return new_states

