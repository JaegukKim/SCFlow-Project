from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

#from SCFlow
def coords_grid_scflow(flow: Tensor) -> Tensor:
    """Generate shifted coordinate grid based based input flow.

    Args:
        flow (Tensor): Estimated optical flow.

    Returns:
        Tensor: The coordinate that shifted by input flow and scale in the
            range [-1, 1].
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(B, 1, 1, 1) + flow
    grid[:, 0, ...] = grid[:, 0, ...] * 2. / max(W - 1, 1) - 1.
    grid[:, 1, ...] = grid[:, 1, ...] * 2. / max(H - 1, 1) - 1.
    grid = grid.permute(0, 2, 3, 1)
    return grid

def filter_flow_by_mask_scflow(flow, gt_mask, invalid_num=400, mode='bilinear', align_corners=False):
    '''Check if flow is valid. 
    When the flow pointed point not in the target image mask or falls out of the target image, the flow is invalid.
    Args:
        flow (tensor): flow from source image to target image, shape (N, 2, H, W)
        mask (tensor): mask of the target image, shape (N, H, W)
    '''
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    mask = gt_mask[:, None].to(flow.dtype)
    grid = coords_grid_scflow(flow)
    mask = F.grid_sample(
        mask,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners
    )
    not_valid_mask = (mask < 0.9) | not_valid_mask[:, None]
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def coords_grid(flow: Tensor) -> Tensor:
    """Generate shifted coordinate grid based based input flow.

    Args:
        flow (Tensor): Estimated optical flow.

    Returns:
        Tensor: The coordinate that shifted by input flow and scale in the
            range [-1, 1].
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(B, 1, 1, 1) + flow
    grid[:, 0, ...] = grid[:, 0, ...] * 2. / max(W - 1, 1) - 1.
    grid[:, 1, ...] = grid[:, 1, ...] * 2. / max(H - 1, 1) - 1.
    grid = grid.permute(0, 2, 3, 1)
    return grid

def filter_flow_by_mask(flow, valid_mask, invalid_num=400, mode='bilinear', align_corners=False):
    '''
    get valid mask
    make invalid pixels invalid num
    '''

    valid_mask = valid_mask[:, None].to(flow.dtype)
    not_valid_mask = valid_mask < 1
    not_valid_mask = not_valid_mask.expand_as(flow)

    masked_flow = flow.clone()
    masked_flow[not_valid_mask] = invalid_num

    return masked_flow

def get_mask_by_flow(flow, que_mask, que_visib_mask, src_mask, invalid_num=400, mode='bilinear', align_corners=False):

    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    not_valid_mask = not_valid_mask[:,None].to(torch.float)

    q_mask = que_mask[:, None].to(flow.dtype)
    grid = coords_grid(flow)
    q_mask = F.grid_sample(
        q_mask,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners
    )
    q_mask = (q_mask >= 0.5)
    q_mask = (q_mask).to(flow.dtype)

    q_visib_mask = que_visib_mask[:, None].to(flow.dtype)
    grid = coords_grid(flow)
    q_visib_mask = F.grid_sample(
        q_visib_mask,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners
    )
    q_visib_mask = (q_visib_mask >= 0.5)
    q_visib_mask = (q_visib_mask).to(flow.dtype) 

    src_mask = src_mask[:,None]
    mask_self_occ = src_mask * (1 - q_mask)
    mask_external_occ = src_mask * (1-q_visib_mask) * (1-mask_self_occ)


    mask_occ = mask_self_occ + mask_external_occ
    valid_mask = (1 - not_valid_mask) * (1 - mask_occ)
    #not_valid_mask = not_valid_mask.expand_as(flow)
    #flow[not_valid_mask] = invalid_num
    return mask_occ, mask_self_occ, mask_external_occ, not_valid_mask, valid_mask
