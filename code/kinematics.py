import numpy as np
from sympy import sin, cos, pi, nsolve, Symbol
import matplotlib.pyplot as plt
import torch


class link_set():
    def __init__(self, position, direction) -> None:
        self.pos0 = position
        self.dir0 = direction
        self.a1 = 30 * np.pi / 180
        self.a2 = 0 * np.pi / 180
        self.a3 = 0 * np.pi / 180

        self.L1_len = 80.10
        self.L2_len = 20.0
        self.L3_len_1 = 60.0
        self.L3_len_2 = 84.0

        self.pos_set = np.array([self.pos0])
        self.forward_k()

    def forward_k(self):
        pos1 = np.array([
            self.pos0[0],
            self.pos0[1] + np.sin(self.a1) * self.L1_len,
            self.pos0[2] + np.cos(self.a1) * self.L1_len
        ])

        pos2 = np.array([
            pos1[0],
            pos1[1] + np.sin(self.a2) * self.L2_len,
            pos1[2] + np.cos(self.a2) * self.L2_len
        ])

        pos3 = np.array([
            pos2[0] + np.sin(self.a3) * self.L3_len_2,
            pos2[1] - np.cos(self.a2) * self.L3_len_1 + np.sin(self.a2) * self.L3_len_2 * np.cos(self.a3),
            pos2[2] + np.sin(self.a2) * self.L3_len_1 + np.cos(self.a2) * self.L3_len_2 * np.cos(self.a3)
        ])

        pos4 = np.array([
            pos3[0],
            pos3[1] + np.sin(self.a2) * self.L2_len,
            pos3[2] + np.cos(self.a2) * self.L2_len
        ])
        self.pos_set = np.array([self.pos0, pos1, pos2, pos3, pos4])
        self.to_world_frame()

    def to_world_frame(self):
        if self.dir0 == 1:
            for pos in self.pos_set:
                pos[0], pos[1] = -pos[1], -pos[0]
        elif self.dir0 == 2:
            pass
        else:
            for pos in self.pos_set:
                pos[0], pos[1] = pos[1], pos[0]

    def sample_angles(self, angles):
        self.a1 = angles[0]
        self.a2 = angles[1]
        self.a3 = angles[2]
        self.forward_k()
        self.to_world_frame()

    def plot_links(self, ax):
        ax.plot(
            self.pos_set[:, 0],
            self.pos_set[:, 1],
            self.pos_set[:, 2],
        )
        ax.scatter3D(
            self.pos_set[:, 0],
            self.pos_set[:, 1],
            self.pos_set[:, 2],
        )


def robot_fk(legs, controlled_angles, passive_angles):
    for i in range(3):
        legs[i].a1 = controlled_angles[i]
        legs[i].a2 = passive_angles[i]
        legs[i].a3 = passive_angles[3 + i]

        legs[i].forward_k()

    return legs


def plot_robot(legs):
    ax = plt.axes(projection='3d')
    ax.plot([0, 10], [0, 0], [0, 0], "red")  # x axis
    ax.plot([0, 0], [0, 10], [0, 0], "blue")  # y axis
    ax.plot([0, 0], [0, 0], [0, 10], "green")  # z axis
    legs[0].plot_links(ax)
    legs[1].plot_links(ax)
    legs[2].plot_links(ax)
    ax.set_xlim([-150, 150])
    ax.set_ylim([-100, 200])
    ax.set_zlim([0, 300])
    plt.show()


def sample_test():
    a1_range = [30, 80]
    a2_range = [-90, 90]
    a3_range = [-60, 60]

    a1_sample = (torch.rand(3) * (a1_range[1] - a1_range[0]) + a1_range[0]) * np.pi / 180
    a2_sample = (torch.rand(3) * (a2_range[1] - a2_range[0]) + a2_range[0]) * np.pi / 180
    a3_sample = (torch.rand(3) * (a3_range[1] - a3_range[0]) + a3_range[0]) * np.pi / 180

    return a1_sample, a2_sample, a3_sample


def generate_robot():
    leg_1 = link_set(position=np.array([-39.5, 27.9, 40.33]), direction=1)
    leg_2 = link_set(position=np.array([-0.4, 67.0, 40.33]), direction=2)
    leg_3 = link_set(position=np.array([39.5, 27.9, 40.33]), direction=3)

    Robot = [leg_1, leg_2, leg_3]
    return Robot


def check_constrains(robot, input, output):
    # c_a, p_a1, p_a2 = sample_test()
    robot = robot_fk(robot, input, output)
    end_1 = robot[0].pos_set[-1]
    end_2 = robot[1].pos_set[-1]
    end_3 = robot[2].pos_set[-1]
    loss_x = abs(end_1[0] + 50 - end_2[0]) + abs(end_3[0] - 50 - end_2[0])
    loss_y = abs(end_1[1] + 23 - end_2[1]) + abs(end_3[1] + 23 - end_2[1])
    loss_z = abs(end_1[2] - end_2[2]) + abs(end_3[2] - end_2[2])
    loss = loss_x + loss_y + loss_z
    return robot, loss


def check_c(motor_a_batch, passive_a_batch, batch_size=10):
    l1 = torch.tensor(80.1)
    l2 = torch.tensor(20.0)
    l3_1 = torch.tensor(60.0)
    l3_2 = torch.tensor(84.0)
    #
    # -(27.9 + torch.sin(motor_a[0]) * l1 + 2 * torch.sin(passive_a[0]) * l2 - torch.cos(
    #     passive_a[0]) * l3_1 + torch.sin(passive_a[0]) * torch.cos(passive_a[3]) * l3_2),
    # motor_a[0,1,2], angle level1;  passive_a[0,1,2], angle level2, passive_a[3,4,5], angle level3
    batch_loss = torch.tensor([], requires_grad=True)
    for b in range(batch_size):
        motor_a = motor_a_batch[b]
        passive_a = passive_a_batch[b]
        pos1 = torch.tensor([
            -(27.9 + torch.sin(motor_a[0]) * l1 + 2 * torch.sin(passive_a[0]) * l2 - torch.cos(
                passive_a[0]) * l3_1 + torch.sin(passive_a[0]) * torch.cos(passive_a[3]) * l3_2),
            -(-39.5 + torch.sin(passive_a[3]) * l3_2),
            40.33 + torch.cos(motor_a[0]) * l1 + 2 * torch.cos(passive_a[0]) * l2 + torch.sin(
                passive_a[0]) * l3_1 + torch.cos(passive_a[0]) * torch.cos(passive_a[3]) * l3_2
        ], requires_grad=True)

        pos2 = torch.tensor([
            -0.4 + torch.sin(passive_a[4]) * l3_2,
            67.0 + torch.sin(motor_a[1]) * l1 + 2 * torch.sin(passive_a[1]) * l2 - torch.cos(
                passive_a[1]) * l3_1 + torch.sin(passive_a[1]) * torch.cos(passive_a[4]) * l3_2,
            40.33 + torch.cos(motor_a[1]) * l1 + 2 * torch.cos(passive_a[1]) * l2 + torch.sin(
                passive_a[1]) * l3_1 + torch.cos(passive_a[1]) * torch.cos(passive_a[4]) * l3_2
        ], requires_grad=True)

        pos3 = torch.tensor([
            27.9 + torch.sin(motor_a[2]) * l1 + 2 * torch.sin(passive_a[2]) * l2 - torch.cos(
                passive_a[2]) * l3_1 + torch.sin(passive_a[2]) * torch.cos(passive_a[5]) * l3_2,

            39.5 + torch.sin(passive_a[5]) * l3_2,

            40.33 + torch.cos(motor_a[2]) * l1 + 2 * torch.cos(passive_a[2]) * l2 + torch.sin(
                passive_a[2]) * l3_1 + torch.cos(passive_a[2]) * torch.cos(passive_a[5]) * l3_2
        ], requires_grad=True)

        loss_x = torch.abs(pos1[0] + 50 - pos2[0]) + torch.abs(pos3[0] - 50 - pos2[0])
        loss_y = torch.abs(pos1[1] + 23 - pos2[1]) + torch.abs(pos3[1] + 23 - pos2[1])
        loss_z = torch.abs(pos1[2] - pos2[2]) + torch.abs(pos3[2] - pos2[2])
        loss = loss_x + loss_y + loss_z
        batch_loss = torch.cat((batch_loss, torch.tensor([loss])))
    return torch.mean(batch_loss)


if __name__ == "__main__":
    my_robot = generate_robot()

    for i in range(10):
        c_a, p_a1, p_a2 = sample_test()
        # robot = robot_fk(my_robot, c_a, np.concatenate((p_a1, p_a2)))
        robot, lo = check_constrains(my_robot, c_a, np.concatenate((p_a1, p_a2)))
        lo2 = check_c([c_a], [torch.cat((p_a1, p_a2))], batch_size=1)
        print(lo, lo2, lo2.item()-lo)
        plot_robot(robot)
