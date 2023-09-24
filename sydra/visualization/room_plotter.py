import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_room(config):
    with open(config, 'r') as f:
        data = json.load(f)

    for i in range(len(data)):

        room_dims = data[i]["room_dims"]
        s_coordinates = data[i]["source_coordinates"]
        m_coordinates = data[i]["mic_coordinates"][0]
        m_array_centres = data[i]["mic_array_centres"]
        reverberation = data[i]['anechoic']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        room_vertices = [
            [0, 0, 0],
            [0, room_dims[1], 0],
            [room_dims[0], room_dims[1], 0],
            [room_dims[0], 0, 0],
            [0, 0, room_dims[2]],
            [0, room_dims[1], room_dims[2]],
            [room_dims[0], room_dims[1], room_dims[2]],
            [room_dims[0], 0, room_dims[2]]
        ]

        room_walls = [
            [room_vertices[0], room_vertices[1], room_vertices[2], room_vertices[3]],  
            [room_vertices[4], room_vertices[5], room_vertices[6], room_vertices[7]], 
            [room_vertices[0], room_vertices[1], room_vertices[5], room_vertices[4]], 
            [room_vertices[1], room_vertices[2], room_vertices[6], room_vertices[5]], 
            [room_vertices[2], room_vertices[3], room_vertices[7], room_vertices[6]],  
            [room_vertices[3], room_vertices[0], room_vertices[4], room_vertices[7]]  
        ]
        if reverberation:
            ax.add_collection3d(Poly3DCollection(room_walls, facecolors='white', linewidths=1, edgecolors='black', alpha=0))
        else: 
            ax.add_collection3d(Poly3DCollection(room_walls, facecolors='grey', linewidths=1, edgecolors='black', alpha=0.2))

        s_x = [coord[0] for coord in s_coordinates]
        s_y = [coord[1] for coord in s_coordinates]
        s_z = [coord[2] for coord in s_coordinates]

        m_x = [coord[0] for coord in m_coordinates]
        m_y = [coord[1] for coord in m_coordinates]
        m_z = [coord[2] for coord in m_coordinates]

        m_array_x = [coord[0] for coord in m_array_centres]
        m_array_y = [coord[1] for coord in m_array_centres]
        m_array_z = [coord[2] for coord in m_array_centres]
        

        ax.scatter(s_x, s_y, s_z, color='red', s=50, label='source', marker='^')
        ax.scatter(m_x, m_y, m_z, color='black', s=50, label='microphone', marker='.')
        ax.scatter(m_array_x, m_array_y, m_array_z, color='blue', s=100, label='microphone array', marker='D')

        max_x = room_dims[0]
        max_y = room_dims[1]
        max_z = room_dims[2]

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        ax.set_zlim([0, max_z])
        ax.set_xlabel('Width')
        ax.set_ylabel('Length')
        ax.set_zlabel('Height')
        ax.set_title('Simulated Room with Sources and Microphones')
        ax.legend()
        tof_dict = {}
        for i in range(len(s_x)):
            for j in range(len(m_x)):
                ax.plot([s_x[i], m_x[j]], [s_y[i], m_y[j]], [s_z[i], m_z[j]], 'k--')
                dist = np.sqrt((s_x[i] - m_x[j])**2 + (s_y[i] - m_y[j])**2 + (s_z[i] - m_z[j])**2)
                tof = 1000* dist / 346  
                if j not in tof_dict:
                    tof_dict[j] = []
                tof_dict[j].append((i, tof))  # Store source index and ToF
        tof_text = "\n".join([f"Microphone {mic_index} and Source {source_index}, ToF: {tof:.2f} microseconds" for mic_index, values in tof_dict.items() for source_index, tof in values])
        ax.text(0, 0, 0, tof_text, fontsize=8, color='black', zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.7))
        tdoa_dict = {}
        for i in range(len(s_x)):
            tdoa_dict[i]=abs(tof_dict[0][i][1]-tof_dict[1][i][1])
        tdoa_text = "\n".join([f"Source {source_index}, TDoA: {tdoa:.2f} microseconds" for source_index, tdoa in tdoa_dict.items()])
        ax.text(0, 0, max_z, tdoa_text, fontsize=8, color='black', zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.7))
        plt.show()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the room simulated.")
    parser.add_argument("config", help="Path to the meta.json file.")
    args = parser.parse_args()
    plot_room(args.config)
