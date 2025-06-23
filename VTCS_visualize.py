import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_pitch(field_dimen=(47.0, 18.5), linewidth=2):
    """Draw a simple ultimate field."""
    fig, ax = plt.subplots(figsize=(12, 6))
    border_dimen = (3, 3)
    half_pitch_length = field_dimen[0] / 2.0
    half_pitch_width = field_dimen[1] / 2.0
    signs = [-1, 1]
    for s in signs:
        ax.plot([s * 14.5, s * 14.5],
                [-half_pitch_width, half_pitch_width],
                'k', linewidth=linewidth)
        ax.plot([-half_pitch_length, half_pitch_length],
                [s * half_pitch_width, s * half_pitch_width],
                'k', linewidth=linewidth)
        ax.plot([s * half_pitch_length, s * half_pitch_length],
                [-half_pitch_width, half_pitch_width],
                'k', linewidth=linewidth)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-half_pitch_length - border_dimen[0],
                 half_pitch_length + border_dimen[0]])
    ax.set_ylim([-half_pitch_width - border_dimen[1],
                 half_pitch_width + border_dimen[1]])
    ax.set_axisbelow(True)
    ax.set_aspect('equal', adjustable='box')
    return fig, ax


def plot_play(play_data, save_path, fps=15, field_dimen=(47.0, 18.5)):
    """Create a simple animation for a play.

    Parameters
    ----------
    play_data : pandas.DataFrame
        Frame by frame tracking data. Expected columns include
        ``frame``, ``id``, ``class``, ``x``, ``y``, ``vx``, ``vy`` and
        ``holder``.
    save_path : str
        Output mp4 file path. The extension ``.mp4`` will be appended if
        not provided.
    fps : int, optional
        Frames per second of the output video.
    field_dimen : tuple of float, optional
        (length, width) of the field in meters.
    """
    if not save_path.endswith('.mp4'):
        save_path += '.mp4'

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='play', artist='Matplotlib')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig, ax = plot_pitch(field_dimen=field_dimen)

    ids = play_data['id'].unique()
    point_objs = {}
    quiver_objs = {}
    text_objs = {}

    for pid in ids:
        point_objs[pid], = ax.plot([], [], 'o', alpha=0.7)
        quiver_objs[pid] = ax.quiver([], [], [], [], scale_units='inches',
                                    scale=10, width=0.0015, headlength=5,
                                    headwidth=3, alpha=0.7)
        text_objs[pid] = ax.text(0, 0, '', fontsize=6, ha='center', va='center')

    x_scale = field_dimen[0] / 94.0
    y_scale = field_dimen[1] / 37.0

    frames = sorted(play_data['frame'].unique())

    with writer.saving(fig, save_path, dpi=100):
        for frame in frames:
            frame_df = play_data[play_data['frame'] == frame]
            for _, row in frame_df.iterrows():
                pid = int(row['id'])
                x = row['x'] * x_scale - field_dimen[0] / 2.0
                y = field_dimen[1] / 2.0 - row['y'] * y_scale
                vx = row['vx'] * x_scale
                vy = -row['vy'] * y_scale

                alpha = 0.7
                if row['class'] == 'offense':
                    color = 'blue' if row['selected'] else 'dodgerblue'
                    size = 10
                elif row['class'] == 'defense':
                    color = 'red' if row['selected_def'] else 'indianred'
                    size = 10
                elif row['class'] == 'disc':
                    color = 'black'
                    size = 4
                    alpha = 1.0
                else:
                    color = 'gray'
                    size = 8

                point_objs[pid].set_data([x], [y])
                point_objs[pid].set_color(color)
                point_objs[pid].set_markersize(size)
                point_objs[pid].set_alpha(alpha)

                if row['class'] != 'disc':
                    quiver_objs[pid].set_offsets(np.array([[x, y]]))
                    quiver_objs[pid].set_UVC(vx, vy)
                    quiver_objs[pid].set_color(color)
                    quiver_objs[pid].set_alpha(alpha)
                else:
                    quiver_objs[pid].set_offsets(np.array([[np.nan, np.nan]]))
                    quiver_objs[pid].set_UVC(0, 0)

                text_objs[pid].set_position((x, y))
                text_objs[pid].set_text(str(pid) if row['class'] != 'disc' else '')
                text_objs[pid].set_color('black' if color == 'blace' else 'white')
                text_objs[pid].set_fontsize(7)

            writer.grab_frame()

    plt.close(fig)
