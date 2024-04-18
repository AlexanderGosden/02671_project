import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def load_config(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def animate_video(video):

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))

    def animate(frame):
        ax.clear()
        ax.imshow(video[frame], cmap='gray')
        ax.set_title(f'Frame {frame}')

    ani = animation.FuncAnimation(fig, animate, frames=range(len(video)), interval=5, repeat=False)

    return ani
