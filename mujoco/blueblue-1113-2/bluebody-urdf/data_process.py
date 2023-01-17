import matplotlib.pyplot as plt
import numpy as np
from train_model import IKModel
import torch
from scipy import signal


def check_data():
    ik_data = np.loadtxt("data/ik_data.csv")
    print(ik_data.shape)
    pos = ik_data[:, :3]
    ang = ik_data[:, 3:]
    ang[:, 0] = (ang[:, 0] + 1.) / 2.
    ang[:, 1] = (ang[:, 1] + .5) / 1.5
    ang[:, 2] = (ang[:, 2] + 1.) / 2.

    print(min(ang[:,0]), max(ang[:,0]))
    print(min(ang[:,1]), max(ang[:,1]))
    print(min(ang[:,2]), max(ang[:,2]))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pos[:,0],
        pos[:,1],
        pos[:,2],
        c= "b",
        s=0.1
    )
    ax.scatter3D(
        input_pos[:,0],
        input_pos[:,1],
        input_pos[:,2],

        c= "r",
        s=1
    )

    plt.title("Work space and target track of one foot")

    plt.show()

def check_train_log():
    log_data_train = np.loadtxt("./log_02/training_MSE.csv")
    log_data_test = np.loadtxt("./log_02/testing_MSE.csv")
    plt.figure()
    plt.plot(log_data_train, label = "train")
    plt.plot(log_data_test, label = "test")
    plt.legend()
    plt.title("Training and testing loss of IK model")
    plt.show()

def model_IK():
    # plt.figure()
    # plt.scatter(x, y)
    # plt.show()
    

    # plt.figure()
    # plt.plot(input_pos[:,0])
    # plt.plot(input_pos[:,1])
    # plt.plot(input_pos[:,2])
    # plt.show()

    print(input_pos.shape)
    Model = IKModel().to('cpu')
    PATH = "./log_02/best_model_MSE.pt"
    Model = IKModel().to('cpu')
    Model.load_state_dict(torch.load(PATH))
    Model.eval()
    pos_data = torch.from_numpy(input_pos).to("cpu", dtype=torch.float)
    m_angles = Model.forward(pos_data)
    m_angles = m_angles.detach().numpy()
    print(m_angles.shape)

    np.savetxt("./angles/circle02.csv", m_angles)

    plt.figure()
    plt.plot(m_angles[:,0])
    plt.plot(m_angles[:,1])
    plt.plot(m_angles[:,2])
    plt.title("Predicted angles by IK model")
    plt.show()


def check_model():
    """check model with train and data"""
    data = np.loadtxt("data/ik_data.csv")
    print(data.shape)
    input_data = data[:100, :3]
    output_label = data[:100, 3:]

    PATH = "./log_02/best_model_MSE.pt"
    Model = IKModel().to('cpu')
    Model.load_state_dict(torch.load(PATH))
    Model.eval()
    pos_data = torch.from_numpy(input_data).to("cpu", dtype=torch.float)
    m_angles = Model.forward(pos_data)
    m_angles = m_angles.detach().numpy()

    plt.figure()
    # plt.plot(m_angles[:, 0] * (1. + 1) - 1, label="out1")
    plt.plot(m_angles[:, 1] * (1. + 0.5) - 0.5, label="out2")
    # plt.plot(m_angles[:, 2] * (1. + 1) - 1, label="out3")

    # plt.plot(output_label[:, 0], label="label1")
    plt.plot(output_label[:, 1], label="label2")
    # plt.plot(output_label[:, 2], label="label3")

    plt.legend()
    plt.title("model eval")
    plt.show()


def data_smooth(orig_data, window=12, plot_flag=False):
    """smooth the predicted data"""
    f_data=np.stack((
        signal.savgol_filter(orig_data[:, 0], window, 3),
        signal.savgol_filter(orig_data[:, 1], window, 3),
        signal.savgol_filter(orig_data[:, 2], window, 3)
    ), 1)



    if plot_flag:
        plt.figure()
        plt.plot(orig_data[:, 0], label = "orig data 1")
        plt.plot(orig_data[:, 1], label = "orig data 2")
        plt.plot(orig_data[:, 2], label = "orig data 3")

        plt.plot(f_data[:, 0], label = "smoothed data 1")
        plt.plot(f_data[:, 1], label = "smoothed data 2")
        plt.plot(f_data[:, 2], label = "smoothed data 3")
        plt.legend()
        plt.show()

    else:
        return f_data


if __name__ == "__main__":
    y = np.array([np.cos(i/32 * np.pi) for i in range(64)]) * 0.03
    z = np.array([np.sin(i/32 * np.pi) for i in range(64)]) * 0.01 - 0.24
    input_pos = np.stack((np.zeros(64) + 0.06, y, z), 1)
    
    check_data()
    check_train_log()
    model_IK()
    check_model()

    # data_smooth(orig_data=np.loadtxt("./angles/circle01.csv"), plot_flag=True)
