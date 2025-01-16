from bop_toolkit.bop_toolkit_lib.pose_error import te, proj, add, adi
import numpy as np

def RETE(out_RT, gt_RT, ids, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    ids = ids.cpu().numpy()
    re = []
    for i, id in enumerate(ids):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis]
        t_g = gt_RT[i, :3, 3][:, np.newaxis]
        re.append(re(R_e, R_g))
        te.append(te(t_e, t_g))
    return (re, te)

def PROJ(out_RT, gt_RT, points, K, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    error = []
    for i in range(out_RT.shape[0]):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis]
        t_g = gt_RT[i, :3, 3][:, np.newaxis]
        K_ = K[i]
        pts = points[i]
        e = proj(R_e, t_e, R_g, t_g, K_, pts)
        error.append(e)
    return error

def ADD(out_RT, gt_RT, points, scale, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    error = []
    for i in range(out_RT.shape[0]):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis]
        t_g = gt_RT[i, :3, 3][:, np.newaxis]
        pts = points[i]
        e = add(R_e, t_e, R_g, t_g, pts) * scale[i]
        error.append(e)
    return error

def ADDS(out_RT, gt_RT, points, symmetry, scale, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    error = []
    for i in range(out_RT.shape[0]):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis]
        t_g = gt_RT[i, :3, 3][:, np.newaxis]
        pts = points[i]
        if symmetry[i]:
            e = adi(R_e, t_e, R_g, t_g, pts) * scale[i]
        else:
            e = add(R_e, t_e, R_g, t_g, pts) * scale[i]
        error.append(e)
    return error

def ADDSS(out_RT, gt_RT, points, scale, **kwargs):
    out_RT = out_RT.detach().cpu().numpy()
    gt_RT = gt_RT.detach().cpu().numpy()
    error = []
    for i in range(out_RT.shape[0]):
        R_e = out_RT[i, :3, :3]
        R_g = gt_RT[i, :3, :3]
        t_e = out_RT[i, :3, 3][:, np.newaxis]
        t_g = gt_RT[i, :3, 3][:, np.newaxis]
        pts = points[i]
        e = adi(R_e, t_e, R_g, t_g, pts) * scale[i]
        error.append(e)
    return error