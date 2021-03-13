from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import os
import sys
import struct
import numpy as np
from numpy import save
from numpy import load
from PIL import Image
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import cv2
import shutil


# This initializes the environment to interact with
unity_env = UnityEnvironment("./square_env/SquareRoom.exe")
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

# This resets the environment to an initial state and provides an initial observation
# - observations are three numpy arrays 
obs_ego, obs_top, vectorial = env.reset()
# obs_ego is the observation that the robot-camera provides (your data)
# obs_top is a topdown view (debug only)
# vectorial contains non-visual signals:
# -> true/false if wall was hit (can be used)
# -> x-position, y-position, rotation angle (ground truth, debug only)

time_between = 0
def init():
    return [im_left, im_right]

def process_frame(frame):
    """Preprocess a 70x210x3 frame to 35x105x1 grayscale
    
    Arguments: 
        frame: The frame to process. Must have values ranging from 0-1
    Returns: 
        The processed frame
    """
    frame = frame[::2,::2]  # Downsample by 2 in every dimension   
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return frame

def mov_y(delta_pos, delta_x):  
    delta_y = np.sqrt(np.round(np.power(delta_pos,2),5)-np.round(np.power(delta_x,2),5))
    delta_y = (-1)**np.random.randint(1,3)*delta_y
    return delta_y

def convert( data_folder, data_file ):

    # datafile
    datafile = open( data_file, 'wb' )

    # input file sequence
    img_sequence = os.listdir( data_folder )
    img_sequence.sort()

    # data header
    frame = load(data_folder + img_sequence[0])*255
    frame = Image.fromarray(frame)
    # frame = Image.open( (data_folder + img_sequence[0]) )

    color_dim = None
    if frame.mode == 'RGBA' or frame.mode == 'RGB':
        print('Color mode is RGB.')
        color_dim = 3
    elif frame.mode == 'F':
        print('Color mode is greyscale.')
        color_dim = 1
    else:
        print('Error! Unknown color mode \'%s\'!' % frame.mode)
        return False

    header = struct.pack( 'iiii', len(img_sequence), # how many frames in the sequence
                                  frame.size[0],     # width of a single frame
                                  frame.size[1],     # height of a single frame
                                  color_dim )        # color dimension (greyscale/RGB)
    datafile.write( header )

    # file info
    print('Writing data to file \'%s\'.' % (data_file))      # color values are only stored as characters

    # data loop
    cnt = 0.0
    for filename in img_sequence:

        try:
            frame_file = load(data_folder + filename)*255
            frame_file = np.uint8(np.round(frame_file))
            frame_file = Image.fromarray(frame_file)
            # frame_file = Image.open( (data_folder + filename) )
            frame_data = frame_file.load()
        except:
            print('\nfailed', filename)
            sys.exit()

        for y in range(0, frame_file.size[1]):
            for x in range(0, frame_file.size[0]):
                if color_dim == 3:
                    datafile.write( struct.pack('c',bytes([frame_data[x,y][0]])))
                    datafile.write( struct.pack('c',bytes([frame_data[x,y][1]])))
                    datafile.write( struct.pack('c',bytes([frame_data[x,y][2]])))
                else:
                    datafile.write( struct.pack('c',bytes([frame_data[x,y]])))

        # progress
        cnt += 1.0
        done = int( cnt/len(img_sequence)*50.0 )
        sys.stdout.write( '\r' + '[' + '='*done + '-'*(50-done) + ']~[' + '%.2f' % (cnt/len(img_sequence)*100.0) + '%]' )
        sys.stdout.flush()

    print('\nAll done.')

def animate(i):
    # Take a step in the environment providing 3d vector (x-movement, y-movement, rotation angle)
    global last_time
    global collision
    global mem_action
    global radians
    global pos_arr
    global num_frames
    
    current_time = time.time()
    time_between = current_time - last_time
    # print(f"Current FPS {1/time_between:.2f}")
    last_time = current_time
    
    if collision==1.0:
        delta_x = np.random.uniform(-delta_pos, delta_pos)
        delta_y = mov_y(delta_pos,delta_x)
        mem_action = [delta_x,delta_y,radians]
        step_output = env.step(mem_action)
    else:
        step_output = env.step(mem_action)

    obs_ego, obs_top, vectorial = step_output[0]
    obs_ego = process_frame(obs_ego)
    collision = vectorial[0]
    
    # Obtain new position
    new_pos = vectorial[1:3].reshape(1,2)
    pos_arr = np.append(pos_arr, new_pos, axis= 0)
    
    # Save image 
    path = 'images/img_' + str(i).zfill(5)
    save(path, obs_ego)

    im_left.set_array(obs_ego)
    im_right.set_array(obs_top)
    
    # progress
    done = int( i/(num_frames-1)*50.0 )
    sys.stdout.write( '\r' + '[' + '='*done + '-'*(50-done) + ']~[' + '%.2f' % (i/(num_frames-1)*100.0) + '%]')
    sys.stdout.flush()
    
    return [im_left, im_right]

# PARAMETERS: linear and angular position change
delta_pos = 0.12  # Euclidian distance travel every iteration
radians = 0.1     # Radians to turn in every iteration
num_frames = 1000  # Default number of frames to be recorded

for i, arg in enumerate( sys.argv ):
    if arg == 'frames': num_frames = int(sys.argv[i+1])

fail = False
if( os.path.isdir('./images/') == True ):
    shutil.rmtree('./images/')
    
try:	os.mkdir('./images/')
except:	fail = True

if fail:
	print('Error! \'images\' folder could not be created.')
	sys.exit()
    
# Variable initialization
delta_x = 0.0
delta_y = mov_y(delta_pos,delta_x)
mem_action = [delta_x,delta_y,radians]
collision = 0.0
pos_arr = np.zeros((1,2))
last_time = time.time()

fig, ax = plt.subplots(1, 2)
im_left = ax[0].imshow(obs_ego, interpolation='none')
im_right = ax[1].imshow(obs_top, interpolation='none')

print('Starting run with %d frames' % num_frames)

#Start animation
anim = animation.FuncAnimation(fig,
                               animate,
                               init_func=init,
                               frames = num_frames,
                               interval=1,
                               repeat = False)

plt.show()

print('\n'+' Run complete' + '\n')


pos_arr = np.round(pos_arr,2)
pos_arr = np.delete(pos_arr, 0, axis=0)  #Erase first row because it was artificially added at the beginning


save('./experiment/trajectory', pos_arr)
  
# Plot random walk trajectory 
fig = plt.figure()
plt.scatter(pos_arr[:,0],pos_arr[:,1],c='r',s=0.5)
plt.title('Random walk trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-8.5,8.5])
plt.ylim([-8.5,8.5])

plt.savefig('./experiment/random_walk_x' + str(num_frames) + '.png')

print('random_walk_x' + str(num_frames) + '.png saved in folder \'/experiment\''+'\n')

# Save all image data into a single file (needed for posterior training)
if 'norecord' not in sys.argv:
    # convert simulation data
    if os.path.isdir('./images/') == True:
        convert( './images/','./experiment/sequence_data' )
    else:
        print('Folder ./images not found')

pos_arr = np.load('./experiment/trajectory.npy')

#env.close()

