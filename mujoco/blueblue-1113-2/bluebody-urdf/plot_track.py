import matplotlib.pyplot as plt
import numpy as np

y = np.array([np.cos(i/16 * np.pi) for i in range(32)]) * 0.03
z = np.array([np.sin(i/16 * np.pi) for i in range(32)]) * 0.01 - 0.24
input_pos = np.stack((np.zeros(32) + 0.06, y, z), 1)

track_data = np.load("./data/record_pos_fixed.npy")
print(track_data.shape)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(
    track_data[100:,0,0],
    track_data[100:,0,1],
    track_data[100:,0,2],
    c= "r",
    # s=0.1,
    # label="foot1"
    label="result of IK"
)

# ax.scatter3D(
#     track_data[:,1,0],
#     track_data[:,1,1],
#     track_data[:,1,2],
#     c= "g",
#     s=0.1,
    
# )

# ax.scatter3D(
#     track_data[:,2,0],
#     track_data[:,2,1],
#     track_data[:,2,2],
#     c= "b",
#     s=1,
#     label="body_center"
# )

ax.plot3D(
    input_pos[:,0],
    input_pos[:,1],
    input_pos[:,2],

    c= "k",
    # s=1,
    label="input track"
)

ax.set_xlim([-0.06,0.06])
plt.legend()
plt.title("Compare the input track and the Inverse Kinematcis result")
plt.show()