import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time

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

up_limit = 1.5
down_limit = -1.

up_angle = 0.3
down_angle = -0.3

# data.ctrl[0] = down_limit
# data.ctrl[1] = down_limit
# data.ctrl[2] = down_limit

def random_move(old_angles):
    insert_steps = 20
    angles = np.random.rand(3) * (1. + 0.8) - 0.8
    angle_list = np.array(
        [np.linspace(old_angles[0], angles[0], insert_steps),
        np.linspace(old_angles[1], angles[1], insert_steps),
        np.linspace(old_angles[2], angles[2], insert_steps)]
    )
    for i in range(insert_steps):
        data.ctrl[0] = angle_list[0,i]
        data.ctrl[1] = angle_list[1,i]
        data.ctrl[2] = angle_list[2,i]
    return angle_list[:, -1]


i = 0
mtime = 0
dt = 0.001
old_angles = np.array([0,0,0])
ik_data = []
while not glfw.window_should_close(window):
    time_prev = mtime

    insert_steps = 30
    angles = np.array([
        np.random.rand() * (1. + 1) - 1,
        np.random.rand() * (1. + 0.5) - 0.5,
        np.random.rand() * (1. + 1) - 1,
    ])
    angle_list = np.array(
        [np.linspace(old_angles[0], angles[0], insert_steps),
        np.linspace(old_angles[1], angles[1], insert_steps),
        np.linspace(old_angles[2], angles[2], insert_steps)]
    )
    # while (mtime - time_prev < 1.0/120.0):

    for count in range(insert_steps):
        data.ctrl[0] = angle_list[0,count]
        data.ctrl[1] = angle_list[1,count]
        data.ctrl[2] = angle_list[2,count]
        # mtime +=dt
        mj.mj_step(model, data)
        time.sleep(dt)

    old_angles = angle_list[:, -1]
    ik_d = np.concatenate((data.site_xpos[0], data.ctrl[:3]))
    print(i, ik_d)
    ik_data.append(ik_d)

    i +=1
    if len(ik_data) == 10000:
        np.savetxt("./data/ik_data_10000.csv", np.array(ik_data))
        break
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