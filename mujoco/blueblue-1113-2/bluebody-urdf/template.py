import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time
from data_process import data_smooth

xml_path = './urdf/bluebody-urdf.xml' #xml file (assumes this is in the same folder as this file)
simend = 50 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -45
cam.distance = 1
cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

up_limit = 1.5
down_limit = -1.

up_angle = 0.2
down_angle = -0.2

def sin_gait_1(i):
    """this gait will turning arround"""
    step_length = 0.3
    n_perc = 8
    a1 = step_length * np.sin(i * np.pi/n_perc)

    a2 = (up_angle-down_angle) / 2 * np.cos(i * np.pi/n_perc) + (up_angle+down_angle) / 2
    a3 = (up_angle-down_angle) / 2 * np.cos(i * np.pi/n_perc + np.pi) + (up_angle+down_angle) / 2
    
    data.ctrl[0] = a2 + a1
    data.ctrl[1] = a2
    data.ctrl[2] = a2 - a1

    data.ctrl[3] = a3 + a1
    data.ctrl[4] = a3
    data.ctrl[5] = a3 - a1
    # data.ctrl[0] = down_angle
    # data.ctrl[1] = down_angle
    # data.ctrl[2] = down_angle

    # data.ctrl[3] = down_angle
    # data.ctrl[4] = down_angle
    # data.ctrl[5] = down_angle

def sin_gait_2(i):
    """speed"""
    step_length = 0.2
    n_perc = 16
    a1 = step_length * np.sin(i * np.pi/n_perc)

    a2 = (up_angle-down_angle) / 2 * np.cos(i * np.pi/n_perc) + (up_angle+down_angle) / 2
    
    data.ctrl[0] = -a2
    data.ctrl[1] = -a2 + 0.05
    data.ctrl[2] = -a2

    data.ctrl[3] = a2
    data.ctrl[4] = a2 + 0.05
    data.ctrl[5] = a2


def model_ik_control(i, p_data):
    idx1 = i % 32
    # idx2 = (i+32) % 64
    data.ctrl[0] = p_data[31-idx1, 0] * (1. + 1) - 1
    data.ctrl[1] = p_data[31-idx1, 1] * (1. + 0.5) - 0.5
    data.ctrl[2] = p_data[31-idx1, 2] * (1. + 1) - 1

    data.ctrl[3] = (p_data[idx1, 0] * (1. + 1) - 1)
    data.ctrl[4] = (p_data[idx1, 1] * (1. + 0.5) - 0.5)
    data.ctrl[5] = (p_data[idx1, 2] * (1. + 1) - 1)


i = 0
mtime = 0
dt = 0.001
predict_data = np.loadtxt("./angles/circle01.csv")
# sm_data = data_smooth(orig_data=predict_data, window=6, plot_flag=False)
sm_data = predict_data
start_t = time.time()
record_pos = []
while not glfw.window_should_close(window):
    time_prev = mtime

    while (mtime - time_prev < 1.0/60.0):
        if i > 100:
            # sin_gait_1(i)
            model_ik_control(i, sm_data)
            record_pos.append(np.array(data.site_xpos))

            # if i > 600:
            #     np.save("./data/record_pos_fixed.npy", record_pos)

            ## speed
        if i == 600:
            T = time.time() - start_t
            distance = np.sqrt((data.site_xpos[-1,0])**2 + (data.site_xpos[-1,1])**2)
            speed = distance / T
            print("Time spend: ", T)
            print("Distancs: ", distance)
            print("Speed: ", speed)


        # mj.mj_forward(model,data)
        # print(data.site_xpos)
        
        cam.lookat = [round(data.site_xpos[-1,0], 4), round(data.site_xpos[-1,1], 4), 0.0]
        mtime +=dt
        mj.mj_step(model, data)
        time.sleep(0.0001)

    i +=1
    # time_prev = data.time

    # while (data.time - time_prev < 1.0/60.0):
    #     mj.mj_step(model, data)

    # if (data.time>=simend):
    #     break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()