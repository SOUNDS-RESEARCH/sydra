# Credits to David Díaz-Guerra.

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation


def animate_trajectory(theta, phi, srp_maps, fps, DOA=None, DOA_est=None, DOA_srpMax=None, file_name=None):
    fig = plt.figure()

    def animation_function(frame, theta, phi, srp_maps, DOA=None, DOA_est=None, DOA_srpMax=None):
        plt.clf()
        srp_map = srp_maps[frame, :, :]
        if DOA is not None:
            DOA = DOA[:frame+1, :]
        if DOA_est is not None:
            DOA_est = DOA_est[:frame+1, :]
        if DOA_srpMax is not None:
            DOA_srpMax = DOA_srpMax[:frame+1, :]
        plot_srp_map(theta, phi, srp_map, DOA, DOA_est, DOA_srpMax)

    anim = animation.FuncAnimation(fig, animation_function, frames=srp_maps.shape[0], fargs=(
        theta, phi, srp_maps, DOA, DOA_est, DOA_srpMax), interval=1e3/fps, repeat=False)

    plt.show()
    plt.close(fig)
    if file_name is not None:
        anim.save(file_name, fps=fps, extra_args=["-vcodec", "libx264"])


def plot_srp_map(theta, phi, srp_map, DOA=None, DOA_est=None, DOA_srpMax=None):
    theta = theta * 180/np.pi
    phi = phi * 180/np.pi
    theta_step = theta[1] - theta[0]
    phi_step = phi[1] - phi[0]
    plt.imshow(srp_map, cmap="inferno", extent=(
        phi[0]-phi_step/2, phi[-1]+phi_step/2, theta[-1]+theta_step/2, theta[0]-theta_step/2))
    plt.xlabel("Azimuth [º]")
    plt.ylabel("Elevation [º]")

    if DOA is not None:
        if DOA.ndim == 1:
            plt.scatter(DOA[1]*180/np.pi, DOA[0]*180/np.pi, c="r")
        elif DOA.ndim == 2:
            DOA_s = np.split(DOA, (np.abs(np.diff(DOA[:, 1])) > np.pi).nonzero()[
                             0] + 1)  # Split when jumping from -pi to pi
            [plt.plot(DOA_s[i][:, 1]*180/np.pi, DOA_s[i][:, 0]*180/np.pi, 'r')
             for i in range(len(DOA_s))]
            plt.scatter(DOA[-1, 1]*180/np.pi, DOA[-1, 0]*180 / np.pi, c='r')
    if DOA_srpMax is not None:
        if DOA_srpMax.ndim == 1:
            plt.scatter(DOA_srpMax[1] * 180/np.pi,
                        DOA_srpMax[0]*180/np.pi, c='k')
        elif DOA_srpMax.ndim == 2:
            DOA_srpMax_s = np.split(DOA_srpMax, (np.abs(np.diff(DOA_srpMax[:, 1])) > np.pi).nonzero()[
                                    0] + 1)  # Split when jumping from -pi to pi
            [plt.plot(DOA_srpMax_s[i][:, 1]*180 / np.pi, DOA_srpMax_s[i]
                      [:, 0]*180 / np.pi, 'k') for i in range(len(DOA_srpMax_s))]
            plt.scatter(DOA_srpMax[-1, 1]*180 / np.pi,
                        DOA_srpMax[-1, 0]*180 / np.pi, c='k')
    if DOA_est is not None:
        if DOA_est.ndim == 1:
            plt.scatter(DOA_est[1]*180/np.pi, DOA_est[0]*180/np.pi, c='b')
        elif DOA_est.ndim == 2:
            DOA_est_s = np.split(DOA_est, (np.abs(np.diff(DOA_est[:, 1])) > np.pi).nonzero()[
                                 0] + 1)  # Split when jumping from -pi to pi
            [plt.plot(DOA_est_s[i][:, 1]*180 / np.pi, DOA_est_s[i]
                      [:, 0]*180 / np.pi, 'b') for i in range(len(DOA_est_s))]
            plt.scatter(DOA_est[-1, 1]*180 / np.pi,
                        DOA_est[-1, 0]*180 / np.pi, c='b')

    plt.xlim(phi.min(), phi.max())
    plt.ylim(theta.max(), theta.min())
    plt.show()


def plotDOA(t, doa):
    """ Plots the groundtruth DOA
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, doa * 180/np.pi)

    ax.legend(["Elevation", "Azimuth"])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("DOA [º]")

    plt.show()


def plotEstimation(self, legned_loc="best"):
    """ Plots the DOA groundtruth and its estimation.
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    If the scene has the field DOAw_srpMax with the SRP-PHAT estimation, it also plots it.
    """
    fig = plt.figure()
    gs = fig.add_gridspec(7, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(self.t, self.source_signal)
    plt.xlim(self.tw[0], self.tw[-1])
    plt.tick_params(axis="both", which="both", bottom=False,
                    labelbottom=False, left=False, labelleft=False)

    ax = fig.add_subplot(gs[1:, 0])
    ax.plot(self.tw, self.DOAw * 180/np.pi)
    plt.gca().set_prop_cycle(None)
    ax.plot(self.tw, self.DOAw_pred * 180/np.pi, "--")
    if hasattr(self, "DOAw_srpMax"):
        plt.gca().set_prop_cycle(None)
        ax.plot(self.tw, self.DOAw_srpMax * 180 / np.pi, "x", markersize=4)

    plt.legend(["Elevation", "Azimuth"], loc=legned_loc)
    plt.xlabel("time [s]")
    plt.ylabel("DOA [º]")

    silences = self.vad.mean(axis=1) < 2/3
    silences_idx = silences.nonzero()[0]
    start, end = [], []
    for i in silences_idx:
        if not i - 1 in silences_idx:
            start.append(i)
        if not i + 1 in silences_idx:
            end.append(i)
    for s, e in zip(start, end):
        plt.axvspan((s-0.5)*self.tw[1], (e+0.5)
                    * self.tw[1], facecolor="0.5", alpha=0.5)

    plt.xlim(self.tw[0], self.tw[-1])
    plt.show()


def plotMap(self, w_idx):
    """ Plots the SRP-PHAT map of the window w_idx.
    If the scene has the fields DOAw, DOAw_pred, DOAw_srpMax it also plot them.
    """
    maps = np.concatenate((self.maps, self.maps[..., 0, np.newaxis]), axis=-1)
    map = maps[w_idx, ...]

    thetaMax = np.pi / 2 if self.mic_array_geometry.arrayType == "planar" else np.pi
    theta = np.linspace(0, thetaMax, maps.shape[-2])
    phi = np.linspace(-np.pi, np.pi, maps.shape[-1])

    DOA = self.DOAw[w_idx, ...] if hasattr(self, "DOAw") else None
    DOA_pred = self.DOAw_pred[w_idx, ...] if hasattr(
        self, "DOAw_pred") else None
    DOA_srpMax = self.DOAw_srpMax[w_idx, ...] if hasattr(
        self, "DOAw_srpMax") else None

    plot_srp_map(theta, phi, map, DOA, DOA_pred, DOA_srpMax)
