import numpy as np
from collections import defaultdict
import os

# unit of distance -> mm, rotation -> degree
def RETE_02(RETE, **kwargs):
    RE, TE = RETE[:, 0], RETE[:, 1]
    check = [(re < 2) and (te < 20) for re, te in zip(RE, TE)]
    score = np.mean(check) * 100
    return score

def RETE_05(RETE, **kwargs):
    RE, TE = RETE[:, 0], RETE[:, 1]
    check = [(re < 5) and (te < 50) for re, te in zip(RE, TE)]
    score = np.mean(check) * 100
    return score

# unit of distance -> pixel
def PROJ_02(PROJ, **kwargs):
    check = [proj < 2 for proj in PROJ]
    score = np.mean(check) * 100
    return score

def PROJ_05(PROJ, **kwargs):
    check = [proj < 5 for proj in PROJ]
    score = np.mean(check) * 100
    return score

def PROJ_10(PROJ, **kwargs):
    check = [proj < 10 for proj in PROJ]
    score = np.mean(check) * 100
    return score

def ADDS_02(ADDS, diameter, **kwargs):
    check = [adds < 0.02 * dia for adds, dia in zip(ADDS, diameter)]
    score = np.mean(check) * 100
    return score

def ADDS_05(ADDS, diameter, **kwargs):
    check = [adds < 0.05 * dia for adds, dia in zip(ADDS, diameter)]
    score = np.mean(check) * 100
    return score

def ADDS_10(ADDS, diameter, **kwargs):
    check = [adds < 0.10 * dia for adds, dia in zip(ADDS, diameter)]
    score = np.mean(check) * 100
    return score

# from https://github.com/ethnhe/FFB6D/blob/master/ffb6d/utils/basic_utils.py
def ADD_AUC(ADD, max_dis=0.1, **kwargs):
    ADD = [add/1000 for add in ADD]
    D = np.array(ADD)
    D[np.where(D > max_dis)] = np.inf;
    D = np.sort(D)
    n = len(ADD)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100

def ADDS_AUC(ADDS, max_dis=0.1, **kwargs):
    ADDS = [add/1000 for add in ADDS]
    D = np.array(ADDS)
    D[np.where(D > max_dis)] = np.inf;
    D = np.sort(D)
    n = len(ADDS)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100

def ADDSS_AUC(ADDSS, max_dis=0.1, **kwargs):
    ADDSS = [add/1000 for add in ADDSS]
    D = np.array(ADDSS)
    D[np.where(D > max_dis)] = np.inf;
    D = np.sort(D)
    n = len(ADDSS)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    return aps * 100

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def total_metric(obj_id, diameter, adds_1, adds_2, is_save=False, is_print=True, prefix=None):
    metrics = defaultdict(list)
    obj_list = list(set(obj_id))
    for obj in obj_list:
        idx = np.where(np.array(obj_id) == obj)[0]
        obj_adds_1 = [adds_1[i] for i in idx]
        obj_adds_2 = [adds_2[i] for i in idx]
        obj_dia = [diameter[i] for i in idx]
        obj_adds_1_mean = sum([o / d for o, d in zip(obj_adds_1, obj_dia)])/len(obj_adds_1)
        obj_adds_2_mean = sum([o / d for o, d in zip(obj_adds_2, obj_dia)])/len(obj_adds_2)
        obj_adds_1_10 = ADDS_10(obj_adds_1, obj_dia)
        obj_adds_2_10 = ADDS_10(obj_adds_2, obj_dia)
        obj_adds_2_05 = ADDS_05(obj_adds_2, obj_dia)
        obj_adds_2_02 = ADDS_02(obj_adds_2, obj_dia)
        obj_adds_2_auc = ADDS_AUC(obj_adds_2)
        metrics['adds_1_mean'].append(obj_adds_1_mean)
        metrics['adds_2_mean'].append(obj_adds_2_mean)
        metrics["adds_1_10"].append(obj_adds_1_10)
        metrics["adds_2_10"].append(obj_adds_2_10)
        metrics["adds_2_05"].append(obj_adds_2_05)
        metrics["adds_2_02"].append(obj_adds_2_02)
        metrics["adds_2_auc"].append(obj_adds_2_auc)
        if is_save:
            with open(f'{prefix}_metrics.txt', 'a') as f:
                f.write(f'obj : {obj}, ADD-S 0.1 : {obj_adds_2_10:.2f}, ADD-S 0.05 : {obj_adds_2_05:.2f}, ADD-S 0.02 : {obj_adds_2_02:.2f}, ADD-S AUC : {obj_adds_2_auc:.2f}\n')
        if is_print:
            print(f'[obj{obj}] ADD-S 0.1 : {obj_adds_2_10:.2f}, ADD-S 0.05 : {obj_adds_2_05:.2f}, ADD-S 0.02 : {obj_adds_2_02:.2f}, ADD-S AUC : {obj_adds_2_auc:.2f}')
    metrics['adds_1_mean'] = np.mean(metrics['adds_1_mean'])
    metrics['adds_2_mean'] = np.mean(metrics['adds_2_mean'])
    metrics["adds_1_10"] = np.mean(metrics["adds_1_10"])
    metrics["adds_2_10"] = np.mean(metrics["adds_2_10"])
    metrics["adds_2_05"] = np.mean(metrics["adds_2_05"])
    metrics["adds_2_02"] = np.mean(metrics["adds_2_02"])
    metrics["adds_2_auc"] = np.mean(metrics["adds_2_auc"])
    if is_save:
        with open(f'{prefix}_metrics.txt', 'a') as f:
            f.write(f'total_score, ADD-S 0.1 : {metrics["adds_2_10"]:.2f}, ADD-S 0.05 : {metrics["adds_2_05"]:.2f}, ADD-S 0.02 : {metrics["adds_2_02"]:.2f}, ADD-S AUC : {metrics["adds_2_auc"]:.2f}')
    if is_print:
        print(f'ADD-S 0.1 : {metrics["adds_2_10"]:.2f}, ADD-S 0.05 : {metrics["adds_2_05"]:.2f}, ADD-S 0.02 : {metrics["adds_2_02"]:.2f}, ADD-S AUC : {metrics["adds_2_auc"]:.2f}')
    return metrics

def challenge_format(scene_id, im_id, obj_id, RT, score, scale, time):
    R = RT[0, 0, :3, :3].flatten().tolist()
    t = (RT[0, 0, :3, 3] * scale[0]).flatten().tolist()
    result = [{
        'scene_id' : scene_id,
        'im_id': im_id,
        'obj_id': obj_id,
        'score': score,
        # 'score': score[0],
        'R': " ".join([str(r) for r in R]),
        't': " ".join([str(s) for s in t]),
        'time': time
    }]
    return result