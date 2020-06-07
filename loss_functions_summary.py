import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp_summary import inverse_warp, flow_warp, pose2flow
from ssim import ssim
epsilon = 1e-8

def spatial_normalize(disp):
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    return disp
    
def spatial_normalize_mean(disp): 
    
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    
    '''
    _max = disp.max(3)[0].unsqueeze(3).max(2)[0].unsqueeze(2)
    disp = disp / _max
    '''
    
    

    return disp

def spatial_normalize_max(disp): 
    '''
    _mean = disp.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    disp = disp / _mean
    
    '''
    _max = disp.max(3)[0].unsqueeze(3).max(2)[0].unsqueeze(2)
    disp = disp / _max


    return disp

def spatial_normalize_min(flow): 
    
    _min = flow.min(3)[0].unsqueeze(3).min(2)[0].unsqueeze(2)
    # print(_min.size())
    flow = flow / _min
    
    '''
    _max = disp.max(3)[0].unsqueeze(3).max(2)[0].unsqueeze(2)
    disp = disp / _max
    '''
    
    

    return flow

def robust_l1(x, q=0.5, eps=1e-2, compute_type = False, q2=0.4, eps2=1e-2):
    if compute_type:
        x = torch.pow((x.pow(2) + eps), q)
        x = x.mean()
    else :
        x = torch.pow((x.abs()+eps2), q2)
        x = x.mean()
    return x

def robust_l1_per_pix(x, q=0.5, eps=1e-2, compute_type = False, q2=0.4, eps2=1e-2):
    if compute_type:
        x = torch.pow((x.pow(2) + eps), q)
    else :
        x = torch.pow((x.abs()+eps2), q2)
    return x


def photometric_flow_loss(maskp01, tgt_img, ref_imgs, flows, intrinsics, intrinsics_inv, depth, depth_ref, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5, add_maskp01=False, add_occ_mask = False, add_less_than_mean_mask = False):
    def one_scale(mask_p01, depth, depth_ref, explainability_mask, occ_masks, flows):
        # assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()
        # downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        # intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        # intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
        

        for i, ref_img in enumerate(ref_imgs_scaled):# 2桢
            # current_pose = pose[:, i]
            current_flow = flows[i]#[B, 2, H, W]

            ref_img_warped = flow_warp(ref_img, current_flow)#[B, 3, H, W]
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()
            """
            # valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) 
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)
            final_mask = None

            if add_maskp01:
                # fwd_mask_p01, _ = build_rigid_maskp01(tgt_img_scaled, ref_img, depth[:,0], depth_ref[b*i:b*(i+1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
                fwd_mask_p01 = mask_p01[:,i,:,:].unsqueeze(1)

            if add_maskp01:
                # print('depth_ref.size()',i,depth_ref.size())
                # print('depth_ref[b*i:b*(i+1), 0].size()',i,depth_ref[b*i:b*(i+1), 0].size())
                diff = diff * fwd_mask_p01
                ssim_loss = ssim_loss * fwd_mask_p01
                with torch.no_grad(): 
                    if final_mask is None :
                        final_mask =  fwd_mask_p01
                    else :
                        final_mask = fwd_mask_p01 * final_mask

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i,:,:].unsqueeze(1).expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i,:,:].unsqueeze(1).expand_as(ssim_loss)
                with torch.no_grad():
                    if final_mask is None :
                        final_mask =  explainability_mask[:,i:i+1]
                    else :
                        final_mask = explainability_mask[:,i:i+1] * final_mask
                

            if add_occ_mask:
                diff = diff * (1 - occ_masks[:,i:i+1])
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])
                if final_mask is None :
                    final_mask =  (1 - occ_masks[:,i:i+1])
                else :
                    final_mask = (1 - occ_masks[:,i:i+1]) * final_mask
                #final_mask = (1 - occ_masks[:,i:i+1])

            # reconstruction_loss += (1- wssim) * oob_normalization_const * (robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            if add_less_than_mean_mask:
                #reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
                
                
                # with torch.no_grad():
                #     threshold = ((((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3))/ (final_mask.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1)  
                #     treshould_matrix = (1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss 
                #     threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                #     final_mask = threshold_mask * final_mask
                with torch.no_grad():
                    if add_maskp01:
                        threshold =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) ) / ( final_mask.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1 )
                        #threshold_max =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1)).max(2)[0].unsqueeze(2).max(3)[0].unsqueeze(3)
                        treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).sum(1).unsqueeze(1)
                        threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                        # threshold_mask = 1-treshould_matrix/threshold_max
                        final_mask = threshold_mask * final_mask
                        #print(final_mask)
                    else:
                        threshold =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) ) / ( (torch.ones_like(diff)).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1 )
                        #threshold_max =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )).sum(1).unsqueeze(1)).max(2)[0].unsqueeze(2).max(3)[0].unsqueeze(3)
                        treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).sum(1).unsqueeze(1)
                        threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                        #threshold_mask = 1-treshould_matrix/threshold_max
                        final_mask = threshold_mask
                        #print(final_mask)

            if final_mask is not None :
                reconstruction_loss +=( ( (((1- wssim) * robust_l1_per_pix(final_mask * diff, q=qch) + wssim * final_mask *ssim_loss )).sum(1).sum(1).sum(1) )/   (final_mask.sum(1).sum(1).sum(1) + 1)    )    .mean()
                assert((reconstruction_loss == reconstruction_loss).item() == 1)
            else:
                reconstruction_loss += (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim *ssim_loss )).mean()

        return reconstruction_loss


    if type(flows[0]) not in [tuple, list]:
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]
    

    loss = 0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        
        if explainability_mask is not None:
        # occ_mask_at_scale = None
        # one_scale(depth, depth_ref, explainability_mask, occ_masks, flows)
            loss += one_scale(maskp01[i], depth[i],depth_ref[i], explainability_mask[i], occ_mask_at_scale, flow_at_scale)
        else :
            loss += one_scale(maskp01[i], depth[i],depth_ref[i], explainability_mask, occ_mask_at_scale, flow_at_scale)
    return loss

#loss1
def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, depth_ref, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5, add_maskp01=False, add_occ_mask = False, add_less_than_mean_mask = False):
    def one_scale(depth, depth_ref, explainability_mask, occ_masks):
        # assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)



        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()
            """
            # valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped)
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)
            final_mask = None
            if add_maskp01:
                fwd_mask_p01, _,_,_,_ = build_rigid_maskp01(tgt_img_scaled, ref_img, depth[:,0], depth_ref[b*i:b*(i+1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            if add_maskp01:
                # print('depth_ref.size()',i,depth_ref.size())
                # print('depth_ref[b*i:b*(i+1), 0].size()',i,depth_ref[b*i:b*(i+1), 0].size())
                diff = diff * fwd_mask_p01
                ssim_loss = ssim_loss * fwd_mask_p01
                with torch.no_grad():
                    if final_mask is None :
                        final_mask = fwd_mask_p01
                    else :
                        final_mask = fwd_mask_p01 * final_mask
                # final_mask =  fwd_mask_p01
            if explainability_mask is not None:
                if explainability_mask[:,i:i+1].sum(1).sum(1).sum(1).min().item() > 0.5:
                    diff = diff *  explainability_mask[:,i:i+1].expand_as(diff)
                    ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)
                    with torch.no_grad():
                        if final_mask is None :
                            final_mask = explainability_mask[:,i:i+1]
                        else :
                            final_mask = explainability_mask[:,i:i+1] * final_mask

                # final_mask = explainability_mask
            
                
            if add_occ_mask:
                diff = diff * (1 - occ_masks[:,i:i+1])
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])
                # final_mask = (1 - occ_masks[:,i:i+1])
                with torch.no_grad():
                    if final_mask is None :
                        final_mask = (1 - occ_masks[:,i:i+1])
                    else :
                        final_mask = (1 - occ_masks[:,i:i+1]) * final_mask
            
            if add_less_than_mean_mask:
                #reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
                
                with torch.no_grad():
                    if final_mask is not None:
                        threshold =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) ) / ( final_mask.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1 )
                        #threshold_max =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* final_mask).sum(1).unsqueeze(1)).max(2)[0].unsqueeze(2).max(3)[0].unsqueeze(3)
                        treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss)
                        #treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).sum(1).unsqueeze(1)
                        threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                        #threshold_mask = 1-treshould_matrix/threshold_max
                        final_mask = threshold_mask * final_mask
                    #print(final_mask)

                    else :
                        threshold =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) ) / ( (torch.ones_like(diff)).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) + 1 )
                        #threshold_max =( (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )).sum(1).unsqueeze(1)).max(2)[0].unsqueeze(2).max(3)[0].unsqueeze(3)
                        treshould_matrix = ((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss).sum(1).unsqueeze(1)
                        threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
                        #threshold_mask = 1-treshould_matrix/threshold_max
                        final_mask = threshold_mask

                #if final_mask.sum(1).sum(1).sum(1).min().item() > 0.5:
                    # print('okokokokok')
                reconstruction_loss += ( ((((1- wssim) * robust_l1_per_pix(final_mask * diff, q=qch) + wssim * final_mask *ssim_loss )).sum(1).sum(1).sum(1) )/ ( final_mask.sum(1).sum(1).sum(1) + 1)).mean()
                assert((reconstruction_loss == reconstruction_loss).item() == 1)
                
                    
            else:
                if final_mask is not None :
                    reconstruction_loss += ( ((((1- wssim) * robust_l1_per_pix(final_mask * diff, q=qch) + wssim * final_mask *ssim_loss )).sum(1).sum(1).sum(1) )/ ( final_mask.sum(1).sum(1).sum(1) + 1)).mean()

                else:
                    reconstruction_loss +=  (((1- wssim) * robust_l1_per_pix( diff, q=qch) + wssim * ssim_loss )).mean()

            #weight /= 2.83
        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0

    '''
    if explainability_mask[0] is not None:
        for d, depth_ref, mask in zip(depth, depth_ref, explainability_mask):
            occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
            loss += one_scale(d, depth_ref, mask, occ_masks)
    else :
        for d,depth_ref in  zip (depth, depth_ref):
            occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
            loss += one_scale(d, depth_ref, None, occ_masks)

    '''
    if explainability_mask[0] is not None:
        occ_masks = depth_occlusion_masks(depth[0], pose, intrinsics, intrinsics_inv)
        loss += one_scale(depth[0], depth_ref[0], explainability_mask[0], occ_masks)
    else :
        occ_masks = depth_occlusion_masks(depth[0], pose, intrinsics, intrinsics_inv)
        loss += one_scale(depth[0], depth_ref[0], None, occ_masks)

    
    return loss

def build_rigid_maskp01(tgt_img, ref_img, depth, depth_ref, current_pose,intrinsics_scaled,intrinsics_scaled_inv, rotation_mode, padding_mode, maskp01_duoci=False):

    with torch.no_grad():
        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        _, fwd_mask_p, fwd_mask_0, bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=True, maskp01_duoci=False, maskp01_qian=None)

        _, bwd_mask_p, bwd_mask_0, fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=True, maskp01_duoci=False, maskp01_qian=None)
        if maskp01_duoci == True:
            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)

            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)

            fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
            bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1

            bwd_mask_1 = inverse_warp(ref_img, depth, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=False, maskp01=False, maskp01_duoci=True, maskp01_qian=fwd_mask_p01)
            fwd_mask_1 = inverse_warp(tgt_img, depth_ref, current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode, reverse_pose=True, maskp01=False, maskp01_duoci=True, maskp01_qian=bwd_mask_p01)
        fwd_mask_p = torch.unsqueeze(fwd_mask_p, 1).float()
        bwd_mask_p = torch.unsqueeze(bwd_mask_p, 1).float()
        fwd_mask_0 = torch.unsqueeze(fwd_mask_0, 1).float()
        bwd_mask_0 = torch.unsqueeze(bwd_mask_0, 1).float()
        fwd_mask_1 = torch.unsqueeze(fwd_mask_1, 1).float()
        bwd_mask_1 = torch.unsqueeze(bwd_mask_1, 1).float()

        fwd_mask_p01 = fwd_mask_p * fwd_mask_0 * fwd_mask_1
        bwd_mask_p01 = bwd_mask_p * bwd_mask_0 * bwd_mask_1
        fwd_mask_p1 = fwd_mask_p *  fwd_mask_1
        bwd_mask_p1 = bwd_mask_p *  bwd_mask_1
        visual_mask =  torch.cat((bwd_mask_p01,fwd_mask_p01,bwd_mask_p1,fwd_mask_p1,bwd_mask_p,fwd_mask_p,bwd_mask_0,fwd_mask_0,bwd_mask_1,fwd_mask_1),1)
    return fwd_mask_p01, bwd_mask_p01,fwd_mask_p1,bwd_mask_p1, visual_mask


def depth_occlusion_masks(depth, pose, intrinsics, intrinsics_inv):
    flow_cam = [pose2flow(depth.squeeze(), pose[:,i], intrinsics, intrinsics_inv) for i in range(pose.size(1))]
    
    if False:
        masks1, masks2 = occlusion_masks(flow_cam[1], flow_cam[2])
        masks0, masks3 = occlusion_masks(flow_cam[0], flow_cam[3])
        masks = torch.stack((masks0, masks1, masks2, masks3), dim=1)
    else:
        masks1, masks2 = occlusion_masks(flow_cam[0], flow_cam[1])
        masks = torch.stack(( masks1, masks2 ), dim=1)
    return masks

def gaussian_explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        loss += torch.exp(-torch.mean((mask_scaled-0.5).pow(2))/0.15)
    return loss


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)

def consensus_exp_masks(cam_flows_fwd, cam_flows_bwd, flows_fwd, flows_bwd, tgt_img, ref_img_fwd, ref_img_bwd, wssim, wrig, ws=0.1):
    exp_masks_target_fwd = []
    exp_masks_target_bwd = []
    def one_scale(cam_flow_fwd, cam_flow_bwd, flow_fwd, flow_bwd, tgt_img, ref_img_fwd, ref_img_bwd, ws):
        b, _, h, w = cam_flow_fwd.size()
        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_img_scaled_fwd = nn.functional.adaptive_avg_pool2d(ref_img_fwd, (h, w))
        ref_img_scaled_bwd = nn.functional.adaptive_avg_pool2d(ref_img_bwd, (h, w))

        cam_warped_im_fwd = flow_warp(ref_img_scaled_fwd, cam_flow_fwd)
        cam_warped_im_bwd = flow_warp(ref_img_scaled_bwd, cam_flow_bwd)

        flow_warped_im_fwd = flow_warp(ref_img_scaled_fwd, flow_fwd)
        flow_warped_im_bwd = flow_warp(ref_img_scaled_bwd, flow_bwd)

        valid_pixels_cam_fwd = 1 - (cam_warped_im_fwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_fwd)
        valid_pixels_cam_bwd = 1 - (cam_warped_im_bwd == 0).prod(1, keepdim=True).type_as(cam_warped_im_bwd)
        # valid_pixels_cam = logical_or(valid_pixels_cam_fwd, valid_pixels_cam_bwd)  # if one of them is valid, then valid

        valid_pixels_flow_fwd = 1 - (flow_warped_im_fwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_fwd)
        valid_pixels_flow_bwd = 1 - (flow_warped_im_bwd == 0).prod(1, keepdim=True).type_as(flow_warped_im_bwd)
        # valid_pixels_flow = logical_or(valid_pixels_flow_fwd, valid_pixels_flow_bwd)  # if one of them is valid, then valid

        cam_err_fwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_fwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_fwd)).mean(1, keepdim=True)) * valid_pixels_cam_fwd
        cam_err_bwd = ((1-wssim)*robust_l1_per_pix(tgt_img_scaled - cam_warped_im_bwd).mean(1,keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, cam_warped_im_bwd)).mean(1, keepdim=True)) * valid_pixels_cam_bwd
        # cam_err = torch.min(cam_err_fwd, cam_err_bwd) * valid_pixels_cam

        flow_err_fwd = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_fwd).mean(1, keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_fwd)).mean(1, keepdim=True) * valid_pixels_flow_fwd
        flow_err_bwd = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_bwd).mean(1, keepdim=True) \
                    + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_bwd)).mean(1, keepdim=True) * valid_pixels_flow_bwd
        # flow_err_bwd = (1-wssim)*robust_l1_per_pix(tgt_img_scaled - flow_warped_im_bwd).mean(1, keepdim=True) \
        #             + wssim*(1 - ssim(tgt_img_scaled, flow_warped_im_bwd)).mean(1, keepdim=True)
        # flow_err = torch.min(flow_err_fwd, flow_err_bwd)

        exp_masks_target_fwd.append( (wrig*cam_err_fwd <= (flow_err_fwd+epsilon)).type_as(cam_err_fwd) )
        exp_masks_target_bwd.append( (wrig*cam_err_bwd <= (flow_err_bwd+epsilon)).type_as(cam_err_bwd) )


    
    for i in range(len(cam_flows_fwd)):
        one_scale(cam_flows_fwd[i], cam_flows_bwd[i], flows_fwd[i], flows_bwd[i], tgt_img, ref_img_fwd, ref_img_bwd, ws)
        ws = ws / 2.3

    return exp_masks_target_fwd , exp_masks_target_bwd

def compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd, THRESH):
    joint_masks = []
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        rigidity_mask_fwd_one_scale = (rigidity_mask_fwd[i] > THRESH).type_as(exp_mask_one_scale)
        rigidity_mask_bwd_one_scale = (rigidity_mask_bwd[i] > THRESH).type_as(exp_mask_one_scale)
        exp_mask_one_scale_joint = 1 - (1-exp_mask_one_scale[:,1])*(1-exp_mask_one_scale[:,2]).unsqueeze(1) > 0.5
        joint_mask_one_scale_fwd = logical_or(rigidity_mask_fwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_bwd = logical_or(rigidity_mask_bwd_one_scale.type_as(exp_mask_one_scale), exp_mask_one_scale_joint.type_as(exp_mask_one_scale))
        joint_mask_one_scale_fwd = Variable(joint_mask_one_scale_fwd.data, requires_grad=False)
        joint_mask_one_scale_bwd = Variable(joint_mask_one_scale_bwd.data, requires_grad=False)
        joint_mask_one_scale = torch.cat((joint_mask_one_scale_bwd, joint_mask_one_scale_bwd,
                        joint_mask_one_scale_fwd, joint_mask_one_scale_fwd), dim=1)
        joint_masks.append(joint_mask_one_scale)

    return joint_masks

def consensus_depth_flow_mask(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce):
    # Loop over each scale
    assert(len(explainability_mask)==len(census_mask_bwd))
    assert(len(explainability_mask)==len(census_mask_fwd))
    loss = 0.
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        census_mask_fwd_one_scale = (census_mask_fwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)
        census_mask_bwd_one_scale = (census_mask_bwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)

        #Using the pixelwise consensus term
        exp_fwd_target_one_scale = exp_masks_fwd_target[i]
        exp_bwd_target_one_scale = exp_masks_bwd_target[i]
        census_mask_fwd_one_scale = logical_or(census_mask_fwd_one_scale, exp_fwd_target_one_scale)
        census_mask_bwd_one_scale = logical_or(census_mask_bwd_one_scale, exp_bwd_target_one_scale)

        # OR gate for constraining only rigid pixels
        # exp_mask_fwd_one_scale = (exp_mask_one_scale[:,2].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # exp_mask_bwd_one_scale = (exp_mask_one_scale[:,1].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # census_mask_fwd_one_scale = 1- (1-census_mask_fwd_one_scale)*(1-exp_mask_fwd_one_scale)
        # census_mask_bwd_one_scale = 1- (1-census_mask_bwd_one_scale)*(1-exp_mask_bwd_one_scale)

        census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
        census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)

        rigidity_mask_combined = torch.cat(( census_mask_bwd_one_scale, census_mask_fwd_one_scale), dim=1)
        loss += weighted_binary_cross_entropy(exp_mask_one_scale, rigidity_mask_combined.type_as(exp_mask_one_scale), [wbce, 1-wbce])

    return loss

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output + epsilon)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))

def edge_aware_smoothness_per_pixel(img, pred):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    pred_gradients_x = gradient_x(pred)
    pred_gradients_y = gradient_y(pred)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y

    return smoothness_x + smoothness_y



def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    '''
    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)

    '''

    scaled_disp = pred_disp[0]
    b, _, h, w = scaled_disp.size()
    scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
    loss += get_edge_smoothness(scaled_img, scaled_disp)


    return loss



def smooth_loss(pred_disp):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        dx, dy = gradient(scaled_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # 2sqrt(2)
    return loss

def occlusion_masks(flow_bw, flow_fw):
    mag_sq = flow_fw.pow(2).sum(dim=1) + flow_bw.pow(2).sum(dim=1)
    #flow_bw_warped = flow_warp(flow_bw, flow_fw)
    #flow_fw_warped = flow_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw
    flow_diff_bw = flow_bw + flow_fw
    occ_thresh =  0.08 * mag_sq + 1.0
    occ_fw = flow_diff_fw.sum(dim=1) > occ_thresh
    occ_bw = flow_diff_bw.sum(dim=1) > occ_thresh
    return occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)
#    return torch.stack((occ_bw.type_as(flow_bw), occ_fw.type_as(flow_fw)), dim=1)

def flow_diff(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear',align_corners=True)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    diff = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    return diff


def compute_epe(gt, pred):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear',align_corners=True)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))

    if nc == 3:
        valid = gt[:,2,:,:]
        epe = epe * valid
        avg_epe = epe.sum()/(valid.sum() + epsilon)
    else:
        avg_epe = epe.sum()/(bs*h_gt*w_gt)

    if type(avg_epe) == Variable: avg_epe = avg_epe.data

    return avg_epe.item()

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear',align_corners=True)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon);
    if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()

def compute_all_epes(gt, rigid_pred, non_rigid_pred, rigidity_mask, THRESH=0.5):
    _, _, h_pred, w_pred = rigid_pred.size()
    _, _, h_gt, w_gt = gt.size()
    rigidity_pred_mask = nn.functional.interpolate(rigidity_mask, size=(h_pred, w_pred), mode='bilinear',align_corners=True)
    rigidity_gt_mask = nn.functional.interpolate(rigidity_mask, size=(h_gt, w_gt), mode='bilinear',align_corners=True)

    non_rigid_pred = (rigidity_pred_mask<=THRESH).type_as(non_rigid_pred).expand_as(non_rigid_pred) * non_rigid_pred
    rigid_pred = (rigidity_pred_mask>THRESH).type_as(rigid_pred).expand_as(rigid_pred) * rigid_pred
    total_pred = non_rigid_pred + rigid_pred

    gt_non_rigid = (rigidity_gt_mask<=THRESH).type_as(gt).expand_as(gt) * gt
    gt_rigid = (rigidity_gt_mask>THRESH).type_as(gt).expand_as(gt) * gt

    all_epe = compute_epe(gt, total_pred)
    rigid_epe = compute_epe(gt_rigid, rigid_pred)
    non_rigid_epe = compute_epe(gt_non_rigid, non_rigid_pred)
    outliers = outlier_err(gt, total_pred)

    return [all_epe, rigid_epe, non_rigid_epe, outliers]


def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]



def new_static_mask( census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH):

    result = []
    with torch.no_grad():
        for i in range(len(census_mask_fwd)): # i 0~5, 6
            exp_fwd_target_one_scale = exp_masks_fwd_target[i]
            exp_bwd_target_one_scale = exp_masks_bwd_target[i]
            
            census_mask_fwd_one_scale = (census_mask_fwd[i] < THRESH).float().prod(dim=1, keepdim=True)
            census_mask_bwd_one_scale = (census_mask_bwd[i] < THRESH).float().prod(dim=1, keepdim=True)
            census_mask_fwd_one_scale = logical_or(census_mask_fwd_one_scale, exp_fwd_target_one_scale)
            census_mask_bwd_one_scale = logical_or(census_mask_bwd_one_scale, exp_bwd_target_one_scale)
            census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
            census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)
        #----------------------------------------------------------------------------------------------------------------------------
            tmp = torch.cat(( census_mask_bwd_one_scale, census_mask_fwd_one_scale), 1)

            result.append( tmp )

    return result

def maskp01_mask(flows_cam_fwd, flow_fwd, flows_cam_bwd, flow_bwd,  tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, depth_ref, pose,rotation_mode='euler', padding_mode='zeros',     THRESH_1=0.01,THRESH_2=0.05):
    '''
    return a list containing 6 mask, whitch size is [B,4,H,W], 0 means bwd-p01, 1 means fwd-p01, 2 means bwd-p1, 3 means fwd-p1
    '''
    result = []
    visual_result = []
    with torch.no_grad():
        for i in range(len(flows_cam_fwd)):
            
            b, _, h, w = depth[i].size()
            downscale = tgt_img.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))


            current_pose = pose[:, 0]

            ref_img_fwd_scaled = nn.functional.adaptive_avg_pool2d(ref_imgs[0], (h, w))           
            bwd_mask_p01, _,bwd_mask_p1,_, visual_mask_bwd = build_rigid_maskp01(tgt_img_scaled, ref_img_fwd_scaled, depth[i][:,0], depth_ref[i][0:b*(1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            #=================================

            current_pose = pose[:, 1]

            ref_img_bwd_scaled = nn.functional.adaptive_avg_pool2d(ref_imgs[1], (h, w))
            fwd_mask_p01, _,fwd_mask_p1,_, _ = build_rigid_maskp01(tgt_img_scaled, ref_img_bwd_scaled, depth[i][:,0], depth_ref[i][b*1:b*(1+1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            
            tmp = torch.cat((bwd_mask_p01,fwd_mask_p01,bwd_mask_p1,fwd_mask_p1),1)


            result.append(tmp)
            visual_result.append(visual_mask_bwd)
            

        return result, visual_result

def a1_a2_rigid_mask(flows_cam_fwd, flow_fwd, flows_cam_bwd, flow_bwd, THRESH_1=0.01,THRESH_2=0.05):
    result = []

    with torch.no_grad():
        for i in range(len(flows_cam_fwd)):
            diff = (flows_cam_fwd[i] - flow_fwd[i]).abs()
            rigid_mask = (torch.pow(diff[:,0],2) + torch.pow(diff[:,1] , 2)) < ( THRESH_1 *( flows_cam_fwd[i].pow(2).sum(dim=1) + flow_fwd[i].pow(2).sum(dim=1)) + THRESH_2 ).type_as(diff)
            tmp_1 = rigid_mask.unsqueeze(1)

            diff = (flows_cam_bwd[i] - flow_bwd[i]).abs()
            rigid_mask = (torch.pow(diff[:,0],2) + torch.pow(diff[:,1] , 2)) < ( THRESH_1 *( flows_cam_bwd[i].pow(2).sum(dim=1) + flow_bwd[i].pow(2).sum(dim=1)) + THRESH_2 ).type_as(diff)
            tmp_2 = rigid_mask.unsqueeze(1)

            tmp = torch.cat((tmp_2,tmp_1),1)

            result.append( tmp.float())
    return result

def less_than_mean_flow_mask(tgt_img, ref_imgs, flows, explainability_mask, lambda_oob=0, qch=0.5, wssim=0.5):#loss4
    def one_scale(explainability_mask, occ_masks, flows):
        # assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        b, _, h, w = flows[0].size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        

        for i, ref_img in enumerate(ref_imgs_scaled):# 2桢
            current_flow = flows[i]#[B, 2, H, W]

            ref_img_warped = flow_warp(ref_img, current_flow)#[B, 3, H, W]
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped)
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i,:,:].unsqueeze(1).expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i,:,:].unsqueeze(1).expand_as(ssim_loss)

            if occ_masks is not None:
                diff = diff *(1-occ_masks[:,i,:,:].unsqueeze(1)).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i,:,:].unsqueeze(1)).expand_as(ssim_loss)

    
            threshold = (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* valid_pixels).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) / valid_pixels.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3)   

            treshould_matrix = (1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss 
            threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
            
            return threshold_mask * valid_pixels


    if type(flows[0]) not in [tuple, list]:
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]
    
    for i in range(0,1):
        flow_at_scale = [uv[i] for uv in flows]
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        
        if explainability_mask is not None:
        # occ_mask_at_scale = None
            return one_scale(explainability_mask[i], occ_mask_at_scale, flow_at_scale)
        else :
            return one_scale(explainability_mask, occ_mask_at_scale, flow_at_scale)
    
def less_than_mean_depth_mask(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros', lambda_oob=0, qch=0.5, wssim=0.5):
    def one_scale(depth, explainability_mask, occ_masks):
        # assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))


        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)


        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()
            """
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) 
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)

            

            if explainability_mask is not None:
                diff = diff * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * (1 - occ_masks[:,i:i+1])* explainability_mask[:,i:i+1].expand_as(ssim_loss)
            else:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            #reconstruction_loss +=  (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            threshold = (((1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss )* valid_pixels).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) / valid_pixels.sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3)   
            treshould_matrix = (1- wssim) * robust_l1_per_pix(diff, q=qch) + wssim * ssim_loss 
            threshold_mask = torch.where(treshould_matrix < threshold, torch.ones_like(treshould_matrix), torch.zeros_like(treshould_matrix))
            return threshold_mask*valid_pixels
            

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    if explainability_mask[0] is not None:
        for d, mask in zip(depth, explainability_mask):
            occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
            return one_scale(d, mask, occ_masks)
    else :
        for d in  depth:
            occ_masks = depth_occlusion_masks(d, pose, intrinsics, intrinsics_inv)
            return one_scale(d, None, occ_masks)

def mask_p01_tensorboard(flows_cam_fwd, flow_fwd, flows_cam_bwd, flow_bwd,  tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, depth_ref, pose,rotation_mode='euler', padding_mode='zeros',     THRESH_1=0.01,THRESH_2=0.05):
    result = []
    with torch.no_grad():
        for i in range(len(flows_cam_fwd)):
            
            b, _, h, w = depth[i].size()
            downscale = tgt_img.size(2)/h
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))


            current_pose = pose[:, 0]

            ref_img_fwd_scaled = nn.functional.adaptive_avg_pool2d(ref_imgs[0], (h, w))           
            bwd_mask_p01, _,bwd_mask_p1,_,_ = build_rigid_maskp01(tgt_img_scaled, ref_img_fwd_scaled, depth[i][:,0], depth_ref[i][0:b*(1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            #=================================

            current_pose = pose[:, 1]

            ref_img_bwd_scaled = nn.functional.adaptive_avg_pool2d(ref_imgs[1], (h, w))
            fwd_mask_p01, _,fwd_mask_p1,_,_ = build_rigid_maskp01(tgt_img_scaled, ref_img_bwd_scaled, depth[i][:,0], depth_ref[i][b*1:b*(1+1), 0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)

            
            tmp = torch.cat((bwd_mask_p01,fwd_mask_p01,bwd_mask_p1,fwd_mask_p1),1)

            result.append(tmp)
            

        return result
    
    def visual(flows,mask_p01): #flows [flows_fwd[i], flows_bwd[i]]
        b, _, h, w = flows[0].size()
        # downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        ref_img_warped_fwd = flow_warp(ref_imgs_scaled[0], flows[0])

        # fwd_mask_p01_pyramid = mask_p01[:,0,:,:].unsqueeze(1)
        # bwd_mask_p01_pyramid = mask_p01[:,1,:,:].unsqueeze(1)


        flows_fwd_x = flows[0][:, 0, :, :].unsqueeze(1)# [B,1,H,W]
        flows_fwd_y = flows[0][:, 1, :, :].unsqueeze(1)
        flows_fwd_x_patches = torch.nn.functional.unfold(flows_fwd_x,(PATCH, PATCH)).permute(0,2,1)# [B, (H-2)*(W-2), 9]
        flows_fwd_y_patches = torch.nn.functional.unfold(flows_fwd_y,(PATCH, PATCH)).permute(0,2,1)#

        flows_fwd_x_patches_middle = flows_fwd_x_patches[:,:, int((PATCH * PATCH - 1) / 2)].unsqueeze(2) # [B, (H-2)*(W-2), 1]
        flows_fwd_y_patches_middle = flows_fwd_y_patches[:,:, int((PATCH * PATCH - 1) / 2)].unsqueeze(2)

        fwd_full_flow_deta_old = -flows_fwd_x_patches_middle * flows_fwd_y_patches + flows_fwd_x_patches * flows_fwd_y_patches_middle
        #[B, (H-2)*(W-2), 9] 

        #===========================================================================================================
        ones_like_deta = torch.ones_like(fwd_full_flow_deta_old).float()#[B, (H-2)*(W-2), 9]
        zeros_like_deta = torch.zeros_like(fwd_full_flow_deta_old).float()#[B, (H-2)*(W-2), 9]
        # mini_deta = 1e-8 * ones_like_deta
        mask_fwd_full_flow_deta = torch.where(torch.abs(fwd_full_flow_deta_old) > 1e-8, ones_like_deta, zeros_like_deta)
        
        fwd_full_flow_deta_new = torch.where(torch.abs(fwd_full_flow_deta_old) > 1e-8, fwd_full_flow_deta_old, 1e-8 * ones_like_deta)
        
        del ones_like_deta
        del zeros_like_deta

        #===========================================================================================================
        bs,_,h, w = flows_fwd_x.size()
        flows_fwd_x_generate_grid = flows_fwd_x[:,0,:,:]

        patches_coords_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(flows_fwd_x_generate_grid).expand_as(flows_fwd_x_generate_grid).unsqueeze(1)  # [bs,1, H, W]
        patches_coords_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(flows_fwd_x_generate_grid).expand_as(flows_fwd_x_generate_grid).unsqueeze(1)  # [bs,1, H, W]
        patches_coords_x = torch.nn.functional.unfold(patches_coords_x,( PATCH, PATCH)).permute(0,2,1)#[B, (H-2)*(W-2), 9]
        patches_coords_y = torch.nn.functional.unfold(patches_coords_y,( PATCH, PATCH)).permute(0,2,1)

        patches_coords_x_middle = patches_coords_x[:,:, int((PATCH * PATCH - 1) / 2)].unsqueeze(2)#[B, (H-2)*(W-2), 1]
        patches_coords_y_middle = patches_coords_y[:,:, int((PATCH * PATCH - 1) / 2)].unsqueeze(2)
        patches_coords_x_deta = patches_coords_x - patches_coords_x_middle#[B, (H-2)*(W-2), 9]
        patches_coords_y_deta = patches_coords_y - patches_coords_y_middle
        del patches_coords_x
        del patches_coords_y
        del patches_coords_x_middle
        del patches_coords_y_middle  

        fwd_full_flow_numda = (-patches_coords_x_deta * flows_fwd_y_patches + patches_coords_y_deta * flows_fwd_x_patches) / fwd_full_flow_deta_new
        fwd_full_flow_mu = (patches_coords_y_deta * flows_fwd_x_patches_middle - patches_coords_x_deta * flows_fwd_y_patches_middle) / fwd_full_flow_deta_new
		#[B, (H-2)*(W-2), 9]


        fwd_full_flow_numda_mu_mask =  ( ((fwd_full_flow_numda > 0) & (fwd_full_flow_numda < 1)) \
					& ((fwd_full_flow_mu > 0) & (fwd_full_flow_mu < 1) ) ).float()
        fwd_full_flow_numda_mu_mask = Variable(fwd_full_flow_numda_mu_mask*mask_fwd_full_flow_deta , requires_grad = False)#[B, (H-2)*(W-2), 9]
        fwd_mask_p01 = torch.nn.functional.unfold( mask_p01.unsqueeze(1) , (PATCH, PATCH)).permute(0,2,1)#[B, (H-2)*(W-2), 9]
        fwd_full_flow_numda_mu_p01_mask = Variable(fwd_full_flow_numda_mu_mask*fwd_mask_p01 , requires_grad = False) #[B, (H-2)*(W-2), 9]
        fwd_full_flow_intersect = torch.exp(-(fwd_full_flow_numda - fwd_full_flow_mu)*(fwd_full_flow_numda - fwd_full_flow_mu)/(fwd_full_flow_numda + fwd_full_flow_mu)) * fwd_full_flow_numda_mu_p01_mask * \
                                              compute_intersect_loss_weights(tgt_img_scaled)#$$$$$
        del fwd_full_flow_numda_mu_mask
        del fwd_mask_p01
        del fwd_full_flow_numda
        del fwd_full_flow_mu

        fwd_full_flow_numda_mu_p01_mask_visula = torch.sum(fwd_full_flow_numda_mu_p01_mask, 2, True).view(bs, h-2, w-2, 1) #[B, (H-2)*(W-2), 1]
        fwd_full_flow_intersect_visula = torch.sum(fwd_full_flow_intersect, 2, True).view(bs, h-2, w-2, 1) #[B, (H-2)*(W-2), 1]

        #====================================================================================================================
        
        
        tmp1 = fwd_full_flow_numda_mu_p01_mask_visula.permute(0, 3, 1, 2)
        tmp2 = fwd_full_flow_intersect_visula.permute(0, 3, 1, 2)

        return tmp1,tmp2
        
    #================================================================================
    flow_intersect_loss = 0
    for i in range(len(flows[0])): # len = 6
        flow_at_scale = [ uv[i] for uv in flows ]
        flow_intersect_loss += one_scale(depth[i], depth_ref[i], flow_at_scale, maskp01[i]) 
    
    tmp1, tmp2 = visual([ flows[1][0] ], maskp01[0][:,1,:,:])
    tmp3, tmp4 = visual([ flows[0][0] ], maskp01[0][:,0,:,:])
    
    
    return flow_intersect_loss,tmp1,tmp2, tmp3,tmp4

def depth_supervise_flow(flows_cam_fwd, flow_fwd, flows_cam_bwd, flow_bwd, THRESH_1=0.01,THRESH_2=0.05, maskp01=None, add_a1_a2_mask=False):
    loss = 0

    for i in range(len(flows_cam_fwd)):
        diff_fwd = (flows_cam_fwd[i] - flow_fwd[i]).abs()
        diff_bwd = (flows_cam_bwd[i] - flow_bwd[i]).abs()

        final_mask = None
        if maskp01 is not None:
            if final_mask is not None:
                final_mask = final_mask*(1-maskp01[i])
            else:
                final_mask =(1 - maskp01[i])
        
        if add_a1_a2_mask:
            with torch.no_grad():
                rigid_mask = (torch.pow(diff_fwd[:,0],2) + torch.pow(diff_fwd[:,1] , 2)) < ( THRESH_1 *( flows_cam_fwd[i].pow(2).sum(dim=1) + flow_fwd[i].pow(2).sum(dim=1)) + THRESH_2 ).type_as(diff_fwd)
                tmp_1 = rigid_mask.unsqueeze(1)          
                rigid_mask = (torch.pow(diff_bwd[:,0],2) + torch.pow(diff_bwd[:,1] , 2)) < ( THRESH_1 *( flows_cam_bwd[i].pow(2).sum(dim=1) + flow_bwd[i].pow(2).sum(dim=1)) + THRESH_2 ).type_as(diff_bwd)
                tmp_2 = rigid_mask.unsqueeze(1)
                a1_a2_mask = torch.cat((tmp_1,tmp_2),1)

                del tmp_1
                del tmp_2
                del rigid_mask

            if final_mask is not None:
                final_mask = final_mask*a1_a2_mask
            else:
                final_mask = a1_a2_mask

        # with torch.no_grad():
        #     threshold_bwd = ( final_mask[:,0,:,:].unsqueeze(1) * robust_l1_per_pix(diff_bwd) ).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) / final_mask[:,0,:,:].unsqueeze(1).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3)
        #     threshold_fwd = ( final_mask[:,1,:,:].unsqueeze(1) * robust_l1_per_pix(diff_fwd) ).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3) / final_mask[:,1,:,:].unsqueeze(1).sum(1).unsqueeze(1).sum(2).unsqueeze(2).sum(3).unsqueeze(3)
        #     threshold_matrix_bwd = final_mask[:,0,:,:].unsqueeze(1) * robust_l1_per_pix(diff_bwd) 
        #     threshold_mask_bwd = torch.where(threshold_matrix_bwd < threshold_bwd, torch.ones_like(threshold_matrix_bwd), torch.zeros_like(threshold_matrix_bwd))
        #     threshold_matrix_fwd = final_mask[:,1,:,:].unsqueeze(1) * robust_l1_per_pix(diff_fwd) 
        #     threshold_mask_fwd = torch.where(threshold_matrix_fwd < threshold_fwd, torch.ones_like(threshold_matrix_fwd), torch.zeros_like(threshold_matrix_fwd))

        # loss += torch.sum( threshold_mask_bwd*final_mask[:,0,:,:].unsqueeze(1) * robust_l1_per_pix(diff_bwd) + threshold_mask_fwd*final_mask[:,1,:,:].unsqueeze(1) * robust_l1_per_pix(diff_fwd) )  / (torch.sum( threshold_mask_bwd*final_mask[:,0,:,:].unsqueeze(1))+torch.sum(threshold_mask_fwd*final_mask[:,1,:,:].unsqueeze(1) ))
        loss += torch.sum( final_mask[:,2,:,:].unsqueeze(1) * robust_l1_per_pix(diff_bwd) + final_mask[:,3,:,:].unsqueeze(1) * robust_l1_per_pix(diff_fwd) )  / (torch.sum( final_mask[:,2,:,:].unsqueeze(1))+torch.sum(final_mask[:,3,:,:].unsqueeze(1) ))
    
        return loss
        
        

        
