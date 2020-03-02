from __future__ import division
import torch
from torch.autograd import Variable
from scipy.sparse import coo_matrix



pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    # del (ones)

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) != h or pixel_coords.size(3) != w:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B,  H, W, 2]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def pose_vec2mat_revised(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    b, _, _ = transform_mat.size()
    filler = Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])).type_as(transform_mat).expand(b, 1, 4)

    transform_mat = torch.cat([transform_mat, filler], dim=1)# [B, 4, 4]

    return transform_mat

def flow_warp(img, flow, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        flow: flow map of the target image -- [B, 2, H, W]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'BCHW')
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    grid_tf = torch.stack((X,Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

    return img_tf


def pose2flow(depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode=None):
    """
    Converts pose parameters to rigid optical flow
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)

def flow2oob(flow):
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    oob = (X.abs()>1).add(Y.abs()>1)>0
    return oob

def occlusion_mask(grid, depth):
    check_sizes(img, 'grid', 'BHW2')
    check_sizes(depth, 'depth', 'BHW')

    mask = grid

    return mask



def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', reverse_pose=False, maskp01=False, maskp01_duoci=False, maskp01_qian=None):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat_revised(pose, rotation_mode)  # [B,3,4]
    if reverse_pose:
        pose_mat = torch.inverse(pose_mat)
    # Get projection matrix for tgt camera frame to source pixel frame
    ones = Variable(torch.zeros([batch_size, 3, 1])).type_as(intrinsics)

    intrinsics = torch.cat([intrinsics, ones], dim=2)
    filler = Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]])).type_as(intrinsics).expand(batch_size, 1, 4)
    intrinsics = torch.cat([intrinsics, filler], dim=1)# [B, 4, 4]

    # del (ones)
    # del (filler)
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    # print(src_pixel_coords)
    # coords_x = (img_width - 1) * (src_pixel_coords[:, :, :, 0] / 2.0 + 0.5)
    # coords_y = (img_height - 1) * (src_pixel_coords[:, :, :, 1] / 2.0 + 0.5)
    # print(coords_x.size())
    # print(coords_x)

    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    if maskp01:
        cam_coords_T = cam2cam(cam_coords, pose_mat)
        mask_p, mask0, mask1 = mask_p01(depth, src_pixel_coords, cam_coords_T)
        return projected_img, mask_p, mask0, mask1
    if maskp01_duoci:  #
        mask1 = mask_p01_duoci(src_pixel_coords, maskp01_qian)

        return mask1
    else:
        return projected_img

def cam2cam(cam_coords, pose):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    pose: A transformation matrix -- [B, 3, 4]
  Returns:
    cam_coords_T: [batch, 3, height, width]
  """
  batch, _, height, width = cam_coords.size()
  ones = Variable(torch.ones([batch, 1, height, width])).type_as(cam_coords)
  cam_coords = torch.cat([cam_coords, ones], dim=1).view(batch, 4, height*width)
  # del (ones)
  # cam_coords = cam_coords
  cam_coords_T = pose.bmm(cam_coords)
  cam_coords_T=cam_coords_T.view(batch, 4, height, width)
  # print(cam_coords_T.size())
  # cam_coords_T = cam_coords_T
  return cam_coords_T[:, :3, :, :]

def mask_p01(depth, coords, cam_coords_T):
      b, h, w, _ = coords.size()

      x0 = torch.floor((w - 1) * (coords[:, :, :, 0] / 2.0 + 0.5).float()).float().cuda()
      x1 = (x0 + 1).float().cuda()
      y0 = torch.floor((h - 1) * (coords[:, :, :, 1] / 2.0 + 0.5).float()).float().cuda()
      y1 = (y0 + 1).float().cuda()


      mask_p_luoji = (x0 >= torch.zeros_like(x0).float().cuda())*(x1 <= torch.Tensor([depth.size()[2] - 1]).float().cuda())*(y0 >= torch.zeros_like(x0).float().cuda())*(y1 <= torch.Tensor([depth.size()[1] - 1]).float().cuda())
      mask_p = mask_p_luoji.float()


      euclidean = torch.sqrt(torch.sum(torch.pow(cam_coords_T, 2), 1)).view(b, -1)
      fuer_like = -2 * torch.ones_like(x0).cuda()
      x0 = torch.where(mask_p_luoji, x0, fuer_like)
      y0 = torch.where(mask_p_luoji, y0, fuer_like)
      xy00 = torch.stack([x0, y0], dim=3).cuda()
      for i in range(b):
        euclideani = euclidean[i, :].view(1, -1)
        xy00_batchi = xy00[i, :, :, :].view(-1, 2).int()  
        unique_xy00_batchi, ids = torch.unique(xy00_batchi[:, 0] * w + xy00_batchi[:, 1],return_inverse=True)

        outputs = coo_matrix((torch.squeeze(1.0 / euclideani, 0).cuda().cpu(), (ids.long().cuda().cpu(), torch.arange(0, euclideani.size()[1] ).long().cuda().cpu())),  shape=(unique_xy00_batchi.size()[0], euclideani.size()[1])).max(1)

        outputs = torch.squeeze(torch.from_numpy(outputs.toarray()).cuda())
        zuixiaojuli = torch.unsqueeze(torch.gather(1.0 / outputs, 0, torch.squeeze(ids).cuda()), 0).float()
        mask0 = torch.unsqueeze(torch.where(zuixiaojuli==euclideani,
                            torch.ones_like(zuixiaojuli).cuda(), torch.zeros_like(zuixiaojuli).cuda()).view(h, w), 0)
        xy00_batchi = xy00_batchi.float().cuda()
        xy10_batchi = xy00_batchi + torch.Tensor([1., 0.]).cuda()
        xy01_batchi = xy00_batchi + torch.Tensor([0., 1.]).cuda()
        xy11_batchi = xy00_batchi + torch.Tensor([1., 1.]).cuda()

        touying = torch.cat([xy00_batchi[:, 1] * w + xy00_batchi[:, 0], xy10_batchi[:, 1] * w + xy10_batchi[:, 0], xy01_batchi[:, 1] * w + xy01_batchi[:, 0], xy11_batchi[:, 1] * w + xy11_batchi[:, 0]], dim=0).long()
        mask1 = torch.zeros(h * w).cuda()
        mask1[touying] = 1
        mask1 = torch.unsqueeze(mask1.view(h, w), 0)
        if i == 0:
          mask0_stack = mask0
          mask1_stack = mask1
        else:
          mask0_stack = torch.cat([mask0_stack, mask0], dim=0)
          mask1_stack = torch.cat([mask1_stack, mask1], dim=0)

      return mask_p, mask0_stack, mask1_stack

def mask_p01_duoci(coords, maskp01_qian):

      b, h, w, _ = coords.size()

      x0 = torch.floor((w - 1) * (coords[:, :, :, 0] / 2.0 + 0.5).float()).float().cuda()
      y0 = torch.floor((h - 1) * (coords[:, :, :, 1] / 2.0 + 0.5).float()).float().cuda()

      # maskp_qian_luoji = torch.eq(maskp_qian.cuda(), torch.ones_like(maskp_qian).cuda())
      # mask1_qian_luoji = torch.eq(mask1_qian.cuda(), torch.ones_like(mask1_qian).cuda())
      # mask0_qian_luoji = torch.eq(mask0_qian.cuda(), torch.ones_like(mask0_qian).cuda())

      maskp01_qian_luoji = torch.eq(maskp01_qian.cuda(), torch.ones_like(maskp01_qian).cuda())
      fuer_like = -2 * torch.ones_like(x0).cuda()
      # x0 = torch.where(maskp_qian_luoji, x0, fuer_like)
      # y0 = torch.where(maskp_qian_luoji, y0, fuer_like)
      # x0 = torch.where(mask1_qian_luoji, x0, fuer_like)
      # y0 = torch.where(mask1_qian_luoji, y0, fuer_like)
      # x0 = torch.where(mask0_qian_luoji, x0, fuer_like)
      # y0 = torch.where(mask0_qian_luoji, y0, fuer_like)

      x0 = torch.where(maskp01_qian_luoji, x0, fuer_like)
      y0 = torch.where(maskp01_qian_luoji, y0, fuer_like)
      xy00 = torch.stack([x0, y0], dim=3).cuda()
      for i in range(b):
        xy00_batchi = xy00[i, :, :, :].view(-1, 2).float().cuda() 
        xy10_batchi = xy00_batchi + torch.Tensor([1., 0.]).cuda()
        xy01_batchi = xy00_batchi + torch.Tensor([0., 1.]).cuda()
        xy11_batchi = xy00_batchi + torch.Tensor([1., 1.]).cuda()

        touying = torch.cat([xy00_batchi[:, 1] * w + xy00_batchi[:, 0], xy10_batchi[:, 1] * w + xy10_batchi[:, 0], xy01_batchi[:, 1] * w + xy01_batchi[:, 0], xy11_batchi[:, 1] * w + xy11_batchi[:, 0]], dim=0).long()
        mask1 = torch.zeros(h * w).cuda()
        mask1[touying] = 1
        mask1 = torch.unsqueeze(mask1.view(h, w), 0)
        if i == 0:
          mask1_stack = mask1
        else:
          mask1_stack = torch.cat([mask1_stack, mask1], dim=0)
      return mask1_stack