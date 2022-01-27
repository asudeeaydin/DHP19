#%%Initialzization

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from os.path import join

import keras.backend as K
from keras.models import load_model
import os
from mpl_toolkits.mplot3d import Axes3D


# Load .h5 file containing
def load_file_(filepath):
    if filepath.endswith('.h5'):
        with h5py.File(filepath, 'r') as f_:
            data = (f_[list(f_.keys())[0]])[()]
    else:
        raise ValueError('.h5 required format.')
    return data
# For each joint and timestep, Gaussian blur is used to smooth each heatmap
def decay_heatmap(heatmap, sigma2=4):
    heatmap = cv2.GaussianBlur(heatmap,(0,0),sigma2)
    heatmap /= np.max(heatmap) # to keep the max to 1
    return heatmap

def generate_gt_heatmap(subj, sess, mov, t, P_mat_cam, images_all, vicon_xyz_all, decay_maps_flag):

   # NB: the order of channels in the .aedat file (and in the saved .h5) is different from the camera index.
    # The next cell takes care of this, loading the proper camera projection matrix.

    vicon_xyz = vicon_xyz_all[t]
    image = images_all[t, :, :, ch_idx]

    # Algorithm to convert XYZ Vicon coordinates to UV pixel coordinates
    # use homogeneous coordinates representation to project 3d XYZ coordinates to 2d UV pixel coordinates.
    vicon_xyz_homog = np.concatenate([vicon_xyz, np.ones([1,13])], axis=0)
    coord_pix_all_cam2_homog = np.matmul(P_mat_cam, vicon_xyz_homog)
    coord_pix_all_cam2_homog_norm = coord_pix_all_cam2_homog/coord_pix_all_cam2_homog[-1]
    u = coord_pix_all_cam2_homog_norm[0]
    v = image_h - coord_pix_all_cam2_homog_norm[1] # flip v coordinate to match the image direction

    # mask is used to make sure that pixel positions are in frame range.
    mask = np.ones(u.shape).astype(np.float32)
    mask[u>image_w] = 0
    mask[u<=0] = 0
    mask[v>image_h] = 0
    mask[v<=0] = 0

    # pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # Generate the heatmaps and plot them over the image

    # initialize the heatmaps
    label_heatmaps = np.zeros((image_h, image_w, num_joints))

    k = 2 # constant used to better visualize the joints when not using decay
    for fmidx,pair in enumerate(zip(v,u, mask)):
        if decay_maps_flag:
            if pair[2]==1: # write joint position only when projection within frame boundaries
                label_heatmaps[pair[0],pair[1], fmidx] = 1
                label_heatmaps[:,:,fmidx] = decay_heatmap(label_heatmaps[:,:,fmidx])
        else:
            if pair[2]==1: # write joint position only when projection within frame boundaries
                label_heatmaps[(pair[0]-k):(pair[0]+k+1),(pair[1]-k):(pair[1]+k+1), fmidx] = 1

    return label_heatmaps, image


# path of files generated using matlab
path_ = '/Users/asudeaydin/PycharmProjects/thesis/previous_work/DHP19/constant_count_frames/h5_dataset_7500_events/346x260'
# home directory of the repository
homedir = '/Users/asudeaydin//Resilio Sync/DHP19/'
# camera projection matrices path
P_mat_dir = join(homedir, 'P_matrices/')
# folder to save fenerated videos
video_folder = '/Users/asudeaydin/PycharmProjects/thesis/previous_work/DHP19/videos'

# DAVIS 260x346 camera is used
image_h, image_w, num_joints = 260, 346, 13 # depend on how accumulated frames are generated in Matlab

t  = 19 # timestep of image to plot
subj, sess, mov = 9, 2, 4
decay_maps_flag = True # True to blur heatmaps
ch_idx = 3 # 0 to 3. This is the order of channels in .aedat/.h5

# Which image of the selected recording to use
imgidx = 24

if ch_idx==1:
    P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
elif ch_idx==3:
    P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
elif ch_idx==2:
    P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
elif ch_idx==0:
    P_mat_cam = np.load(join(P_mat_dir,'P4.npy'))

# .h5 are the event frames

# load files of images and labels, and select the single sample t to plot
vicon_xyz_all = load_file_(join(path_, 'S{}_session{}_mov{}_7500events_label.h5'.format(subj,sess,mov)))
images_all = load_file_(join(path_, 'S{}_session{}_mov{}_7500events.h5'.format(subj,sess,mov)))
images_all_shape = np.shape(images_all)
t_end = images_all_shape[0]

# Convert gt heatmaps into video
video_name = 'gt_heatmaps.mp4'
path_video = join(video_folder, video_name)
ims = []
fig = plt.figure()
plt.axis('off')

for time_id in range(t_end):
    label_heatmaps, image = generate_gt_heatmap(subj, sess, mov, time_id, P_mat_cam, images_all, vicon_xyz_all, decay_maps_flag)
    overlay_image = cv2.addWeighted(image, 0.0039, np.sum(label_heatmaps, axis=-1), 2, 0.0)
    im = plt.imshow(overlay_image, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat = False) #20fps

# plt.show()
ani.save(path_video)