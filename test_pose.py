import torch
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import models
from inverse_warp_summary import pose_vec2mat


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--posenet", type=str, default="PoseNetB6", help="PoseNet model path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)


def main():
    args = parser.parse_args()
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

    weights = torch.load(args.pretrained_posenet)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=seq_length - 1).cuda()
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))


        global_pose = np.identity(4) #
        pose_list=[global_pose[0:3,:].reshape(1,12)]



    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs_var = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).cuda()
            img_var = Variable(img, volatile=True) 
            if i == len(imgs)//2:
                tgt_img_var = img_var
            else:
                ref_imgs_var.append(Variable(img, volatile=True))

        if args.posenet in ["PoseNet6", "PoseNetB6"]:
            poses = pose_net(tgt_img_var, ref_imgs_var)
        else:
            _, poses = pose_net(tgt_img_var, ref_imgs_var)

        if j == 0 :
            poses = poses.cpu().data[0]
            #print(poses.size())  #[2,6]
            # poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])
            # print(poses.size()) #[3,6]
            
            pose = poses[0,:].unsqueeze(0)
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()#[B, 3, 4]
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ (pose_mat)


            scale_factor = np.sum(sample['poses'][0,:,-1] * pose_mat[0:3,-1])/np.sum(pose_mat[0:3,-1] ** 2)
            global_pose = np.hstack([global_pose[:,:-1], scale_factor * global_pose[:,-1:]])

            pose_list.append(global_pose[0:3,:].reshape(1,12))

            #------------------------------------------------------
            
            pose = poses[1,:].unsqueeze(0)
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()#[B, 3, 4]
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)

            pose_gt = np.vstack([sample['poses'][1,:,:], np.array([0, 0, 0, 1])])
            scale_factor = np.sum(  np.linalg.inv(pose_gt)[:3,-1] * pose_mat[0:3,-1])/np.sum(pose_mat[0:3,-1] ** 2)
            global_pose = np.hstack([global_pose[:,:-1], scale_factor * global_pose[:,-1:]])
            
            pose_list.append(global_pose[0:3,:].reshape(1,12))

        else :
            poses = poses.cpu().data[0]
            #print(poses.size())  #[2,6]
            # poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])
            # print(poses.size()) #[3,6]
            
            pose = poses[1,:].unsqueeze(0)
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()#[B, 3, 4]
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)

            pose_gt = np.vstack([sample['poses'][1,:,:], np.array([0, 0, 0, 1])])
            scale_factor = np.sum(  np.linalg.inv(pose_gt)[:3,-1] * pose_mat[0:3,-1])/np.sum(pose_mat[0:3,-1] ** 2)
            print(scale_factor)
            global_pose = np.hstack([global_pose[:,:-1], scale_factor * global_pose[:,-1:]])
            
            
            pose_list.append(global_pose[0:3,:].reshape(1,12))

        
    
    pose_list = np.concatenate(pose_list, axis=0)
    filename = Path(args.output_dir  + "09_pred" + ".txt")
    np.savetxt(filename, pose_list, delimiter=' ', fmt='%1.8e')

def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length


if __name__ == '__main__':
    main()
