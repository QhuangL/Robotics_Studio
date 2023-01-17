import numpy as np
import matplotlib.pyplot as plt

def matrix_to_RPY(Rm):
    Rm = Rm.reshape(3, 3)
    yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    pitch = np.arctan2(-Rm[2, 0], np.sqrt(Rm[2, 1] ** 2 + Rm[2, 2] ** 2))
    roll = np.arctan2(Rm[2, 1], Rm[2, 2])
    return np.array([roll, pitch, yaw])

def para_period(para, gait_points=64, check_plot=False):
    X = [i for i in range(gait_points)]
    Y = np.array([
        [para[0, 0] * np.sin((x / 32. + 2 * para[0, 1]) * np.pi) +
         para[0, 2] * np.sin((x / 16. + 2 * para[0, 3]) * np.pi) +
         para[0, 4] * np.sin((x / 8. + 2 * para[0, 5]) * np.pi)
         for x in X],
        [para[1, 0] * np.sin((x / 32. + 2 * para[1, 1]) * np.pi) +
         para[1, 2] * np.sin((x / 16. + 2 * para[1, 3]) * np.pi) +
         para[1, 4] * np.sin((x / 8. + 2 * para[1, 5]) * np.pi)
         for x in X],
        [para[2, 0] * np.sin((x / 32. + 2 * para[2, 1]) * np.pi) +
         para[2, 2] * np.sin((x / 16. + 2 * para[2, 3]) * np.pi) +
         para[2, 4] * np.sin((x / 8. + 2 * para[2, 5]) * np.pi)
         for x in X],
    ]) / 2.

    if check_plot:
        plt.figure()
        plt.plot(Y[0])
        plt.plot(Y[1])
        plt.plot(Y[2])
        plt.show()

    gait_f = np.concatenate((Y, np.flip(Y, 1)))
    return gait_f

def check_para(para_path):
    par = np.loadtxt(para_path)[-1].reshape(3,6)
    para_period(par, 64, True)

if __name__ == "__main__":
    check_para("para_data/gene3.txt")
