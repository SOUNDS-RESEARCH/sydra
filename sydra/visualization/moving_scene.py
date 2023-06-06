import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


def plot_moving_source(room_dims, mic_coordinates, traj_pts, rt60, snr, view="3D", output_filename=None):
    """ Plots the source trajectory and the microphones within the room
    """
    assert view in ["3D", "XYZ", "XY", "XZ", "YZ", "XY-XZ"]

    if view == "3D" or view == "XYZ":  # Plot 3D view
        plot_3d_scene(room_dims, mic_coordinates, traj_pts, rt60, snr)
    elif view == "XY-XZ":
        fig, axs = plt.subplots(ncols=2)
        plot_moving_source_2d_projection(room_dims, mic_coordinates, traj_pts, "XY", axs[0])
        plot_moving_source_2d_projection(room_dims, mic_coordinates, traj_pts, "XZ", axs[1])

    else:
        plot_moving_source_2d_projection(room_dims, mic_coordinates, traj_pts, mode=view)

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()


def plot_3d_scene(room_dims, mic_coordinates, traj_pts, rt60, snr, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)

    # 1. Set figure limits to room dimensions
    ax.set_xlim3d(0, room_dims[0])
    ax.set_ylim3d(0, room_dims[1])
    ax.set_zlim3d(0, room_dims[2])

    # 2. Draw trajectory points
    cmap = matplotlib.cm.get_cmap('BuPu')
    color_grad = [cmap(i) for i in np.linspace(0.2, 1, len(traj_pts))]
    ax.scatter(traj_pts[:, 0], traj_pts[:, 1], traj_pts[:, 2], c=color_grad)

    # 3. Draw microphone centre
    ax.scatter(mic_coordinates[:, 0],
               mic_coordinates[:, 1], mic_coordinates[:, 2])
    ax.text(traj_pts[0, 0], traj_pts[0, 1], traj_pts[0, 2], "start")

    ax.set_title(
        "$T_{60}$" + " = {:.3f}s, snr = {:.1f}dB".format(rt60, snr))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def plot_moving_source_2d_projection(room_dims, mic_coordinates, traj_pts, mode="XY", ax=None):
    "Plot a projection of a 3D scene"

    assert mode in ["XY", "XZ", "YZ"]

    if mode == "XY":
        idxs = [0, 1]
        axis_labels = ["x", "y"]
    elif mode == "XZ":
        idxs = [0, 2]
        axis_labels = ["x", "z"]
    elif mode == "YZ":
        idxs = [1, 2]
        axis_labels = ["y", "z"]
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(0, room_dims[idxs[0]])
    ax.set_ylim(0, room_dims[idxs[1]])

    cmap = matplotlib.cm.get_cmap('BuPu')
    color_grad = [cmap(i) for i in np.linspace(0.2, 1, len(traj_pts))]
    ax.scatter(traj_pts[:, idxs[0]], traj_pts[:, idxs[1]], c=color_grad)
    ax.scatter(mic_coordinates[:, idxs[0]], mic_coordinates[:, idxs[1]])
    # ax.text(traj_pts[0, idxs[0]], traj_pts[0, idxs[1]], "start")
    ax.legend(["Source start", "Microphone array"])
    ax.set_xlabel(f"{axis_labels[0]} [m]")
    ax.set_ylabel(f"{axis_labels[1]} [m]")
    plt.tight_layout()


def plot_srp_animation(srp_maps, mic_array_geometry="planar",
                       doa_w=None, doa_w_pred=None, doa_w_srp_max=None,
                       fps=10, file_name=None):
    """ Creates an animation with the SRP-PHAT maps of each window.
    The scene need to have the field maps with the SRP-PHAT map of each window.
    If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also includes them.
    """
    maps = np.concatenate((srp_maps, srp_maps[..., 0, np.newaxis]), axis=-1)

    theta_max = np.pi/2 if mic_array_geometry == "planar" else np.pi
    theta = np.linspace(0, theta_max, maps.shape[-2])
    phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

    animate_trajectory(theta, phi, maps, fps, doa_w,
                       doa_w_pred, doa_w_srp_max, file_name)
