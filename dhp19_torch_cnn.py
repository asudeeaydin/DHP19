import matplotlib.pyplot as plt
import h5py
import numpy as np
from os.path import join
import torch

def load_projection_matrix(ch_idx):
    # camera projection matrices path
    P_mat_dir = '/Users/asudeaydin/Resilio Sync/DHP19/P_matrices'

    if ch_idx==1:
        P_mat_cam = np.load(join(P_mat_dir,'P1.npy'))
    elif ch_idx==3:
        P_mat_cam = np.load(join(P_mat_dir,'P2.npy'))
    elif ch_idx==2:
        P_mat_cam = np.load(join(P_mat_dir,'P3.npy'))
    elif ch_idx==0:
        P_mat_cam = np.load(join(P_mat_dir,'P4.npy'))
    return P_mat_cam
def pixel_projection(xyz_label):
    vicon_xyz_homog = torch.cat((xyz_label, torch.ones(xyz_label.size()[0], 1, 13)),1)
    pix_homog = torch.einsum('ij, kjm -> kim', P_mat_cam, vicon_xyz_homog)

    u = torch.div(pix_homog[:,0,:], pix_homog[:, -1, :])
    u.masked_fill_(u>image_w, 0)
    v = image_h - torch.div(pix_homog[:,1,:], pix_homog[:, -1, :])
    v.masked_fill_(v>image_h, 0)

    return u.type(torch.int64), v.type(torch.int64)  #int(), v.int()

test_path = '/Users/asudeaydin/PycharmProjects/thesis/previous_work/DHP19/constant_count_frames/h5_dataset_7500_events/346x260/S9_session2_mov4_7500events.h5'
test_label_path = '/Users/asudeaydin/PycharmProjects/thesis/previous_work/DHP19/constant_count_frames/h5_dataset_7500_events/346x260/S9_session2_mov4_7500events_label.h5'
dataset_path = '/Users/asudeaydin/PycharmProjects/thesis/previous_work/DHP19/constant_count_frames/h5_dataset_7500_events/346x260/'

camera_id = 3
P_mat_cam = torch.from_numpy(load_projection_matrix(camera_id))
image_h, image_w, num_joints = 260, 346, 13 # depend on how accumulated frames are generated in Matlab

rec = h5py.File(test_path, 'r')
xyz_label = h5py.File(test_label_path, 'r')

frames = torch.from_numpy(rec['DVS'][:,:,:,camera_id])
labels = torch.from_numpy(xyz_label['XYZ'][:,:,:])

u, v = pixel_projection(labels)
image_id = 200

joint_locs = torch.zeros(image_h, image_w)
joint_locs[v[image_id], u[image_id]] = 1 #image with joint locations

plt.figure()
plt.imshow(joint_locs)
plt.imshow(frames[image_id], cmap='gray', alpha=0.2)
plt.axis('off')
plt.show()
