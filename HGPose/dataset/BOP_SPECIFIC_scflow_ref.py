from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from HGPose.utils.dataset import load_json_fast
from HGPose.utils.geometry import (random_RT, get_ref_RT, RT_from_boxes, positive_RT, bbox_add_noise,
	get_K_crop_resize, squaring_boxes, image_cropping, mesh_convert_coord, default_K, render_geometry,
	farthest_rotation_sampling, carving_feature, R_T_to_RT, render_geometry_scflow, get_flow_from_delta_pose_and_xyz, get_flow_from_delta_pose_and_xyz_scflow, jitter_RT_with_scale)
# from HGPose.utils.geometry import (positive_RT, negative_RT, get_ref_RT, jitter_RT, bbox_add_noise, get_flow_from_delta_pose_and_xyz, apply_forward_flow,
# 	get_K_crop_resize, squaring_boxes, image_cropping, mesh_convert_coord, random_RT, eval_rot_error, carving_feature, 
# 	apply_backward_flow, 
# 	default_K, R_T_to_RT, render_geometry, RT_from_boxes, encode_coordinate)
from HGPose.utils.object import set_object
from HGPose.model.clip import CLIP
from collections import defaultdict
from bop_toolkit.bop_toolkit_lib.inout import load_scene_camera, load_scene_gt, load_json, load_ply
from glob import glob
from tqdm import tqdm
from abc import abstractmethod
from copy import deepcopy
import psutil
import time
import os
from pytorch3d.io import IO
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.renderer import TexturesVertex, Textures
from pytorch3d.ops import sample_farthest_points
import matplotlib.pyplot as plt
import trimesh

def get_dataset(cfg, is_train, is_memory):
    if cfg.dataset_name == 'lmo':
        dataset = lmo_dataset(cfg, is_train, is_memory)
    elif cfg.dataset_name == 'ycbv':
        dataset = ycbv_dataset(cfg, is_train, is_memory)
    return dataset

def get_least_loaded_cores(rank):
    time.sleep(rank)
    cpu_percentages = psutil.cpu_percent(percpu=True, interval=5.0)
    return sorted(range(len(cpu_percentages)), key=lambda x: cpu_percentages[x])

def worker_init_fn(worker_id, least_loaded_cores):
    # 사용량이 가장 낮은 core 선택
    core_to_use = least_loaded_cores[worker_id % len(least_loaded_cores)]
    print(f'worker core: {core_to_use}')
    # 해당 worker의 프로세스에 CPU affinity 설정
    current_process = psutil.Process()
    current_process.cpu_affinity([core_to_use])

class BOP_Dataset(Dataset):
    def __init__(self, cfg, is_train=True, is_memory=False):
        self.num_query_views = cfg.num_query_views
        self.num_input_views = cfg.num_input_views
        self.N_ref = self.num_input_views
        self.image_size = cfg.image_size
        self.ref_size = cfg.ref_size
        self.geo_size = cfg.geo_size

        self.represent_mode = cfg.represent_mode
        self.ray_mode = cfg.ray_mode

        self.obj_list = cfg.obj_list
        self.obj_name_list = cfg.obj_name_list
        self.use_carved = cfg.use_carved
        self.data_domain = cfg.data_domain

        self.is_memory = is_memory
        self.is_train = is_train

        self.carving_size = cfg.carving_size


        

        
    @abstractmethod
    def load_train_ids(self, *args):
        raise NotImplementedError
    
    @abstractmethod
    def load_test_ids(self, *args):
        raise NotImplementedError

    @abstractmethod
    def load_test_bbox(self, *args): 
        raise NotImplementedError

    def load_pbr_ids(self):
        ids = defaultdict(lambda: defaultdict(list))
        scene_folders = glob(f'{self.root_path}/train_pbr/*')
        for scene_folder in scene_folders:
            scene_gt = load_scene_gt(f'{scene_folder}/scene_gt.json')
            im_list = scene_gt.keys()
            for im_id in im_list:
                for inst_id in range(len(scene_gt[im_id])):
                    scene_id = int(scene_folder.split('/')[-1])
                    im_id = int(im_id)
                    ids[scene_id][im_id] += [inst_id]
        return ids

    def load_dataset(self, ids, mode):
        if 'pbr' in mode:
            visible_fract_thr = 0.2
            img_type = 'jpg'
        else:
            visible_fract_thr = 0.0
            img_type = 'png'
        image_data = defaultdict(dict)
        instance_data = defaultdict(lambda: defaultdict(dict))
        for scene_id, im_dict in tqdm(ids.items()):
            scene_str = "{0:0=6d}".format(scene_id)
            gt_path = f'{self.root_path}/{mode}/{scene_str}/scene_gt.json'
            camera_path = f'{self.root_path}/{mode}/{scene_str}/scene_camera.json'
            info_path = f'{self.root_path}/{mode}/{scene_str}/scene_gt_info.json'
            scene_gt = load_scene_gt(gt_path)
            scene_camera = load_scene_camera(camera_path)
            scene_info = load_json(info_path, keys_to_int=True)
            for im_id, inst_list in im_dict.items():
                im_str = "{0:0=6d}".format(im_id)
                camera = scene_camera[im_id]
                if self.is_memory:
                    image = np.array(Image.open(f'{self.root_path}/{mode}/{scene_str}/rgb/{im_str}.{img_type}').convert("RGB"))
                    depth = np.array(Image.open(f'{self.root_path}/{mode}/{scene_str}/depth/{im_str}.png'))
                else:
                    image = f'{self.root_path}/{mode}/{scene_str}/rgb/{im_str}.{img_type}'
                    depth = f'{self.root_path}/{mode}/{scene_str}/depth/{im_str}.png'
                image_data[(scene_id, im_id)] = {
                    'K' : torch.tensor(camera['cam_K']).to(torch.float),
                    'depth_scale' : camera['depth_scale'],
                    'image' : image,
                    'depth': depth,
                    'scene_id' : scene_id,
                    'im_id' : im_id,
                }
                for inst_id in inst_list:
                    inst_str = "{0:0=6d}".format(inst_id)
                    gt = scene_gt[im_id][inst_id]
                    info = scene_info[im_id][inst_id]
                    if gt['obj_id'] in self.obj_list and info['visib_fract'] > visible_fract_thr:
                        info['bbox_obj'][2] += info['bbox_obj'][0]
                        info['bbox_obj'][3] += info['bbox_obj'][1]
                        RT = torch.tensor(R_T_to_RT(gt['cam_R_m2c'], gt['cam_t_m2c'])).to(torch.float)
                        #RT[..., :3, 3] = RT[..., :3, 3] / self.object_param[gt['obj_id']]['scale'] 
                        if self.is_memory:
                            mask = np.asarray(Image.open(f'{self.root_path}/{mode}/{scene_str}/mask/{im_str}_{inst_str}.png'))
                            mask_visib = np.asarray(Image.open(f'{self.root_path}/{mode}/{scene_str}/mask_visib/{im_str}_{inst_str}.png'))
                        else:
                            mask = f'{self.root_path}/{mode}/{scene_str}/mask/{im_str}_{inst_str}.png'
                            mask_visib = f'{self.root_path}/{mode}/{scene_str}/mask_visib/{im_str}_{inst_str}.png'
                        instance_data[(scene_id, im_id)][inst_id]= {
                            'RT' : RT,
                            'bbox_obj' : torch.tensor(info['bbox_obj']).to(torch.float),
                            'visib_fract' : info['visib_fract'],
                            'px_count_visib': info['px_count_visib'],
                            'mask' : mask,
                            'mask_visib' : mask_visib,
                            'inst_id' : inst_id,
                            'obj_id' : gt['obj_id']
                        }
        return {'image_data': image_data, 'instance_data': instance_data}
    
    def set_object_scflow(self, obj_list, include_textures=False):
        object_info = load_json(f'{self.root_path}/models/models_info.json')
        obj_param = defaultdict(dict)
        for obj_id in obj_list:
            info = object_info[str(obj_id)]
            full_mesh = IO().load_mesh(f'{self.root_path}/models/obj_{obj_id:06}.ply', include_textures=True, device='cuda')
            if full_mesh.textures == None:
                                file_list = os.listdir(f'{self.root_path}/models')
                                png_files = [file for file in file_list if file.endswith(".png")]
                                if png_files:
                                    full_mesh = self.get_texture_mesh_with_trimesh(f'{self.root_path}/models/obj_{obj_id:06}.ply')
            mesh = IO().load_mesh(f'{self.root_path}/models_eval/obj_{obj_id:06}.ply', include_textures=False, device='cuda')
            scale = float(torch.norm(mesh.verts_packed(), dim=1, p=2).max())
            full_mesh = full_mesh.scale_verts(1/scale)
            mesh = mesh.scale_verts(1/scale)
            pts = mesh.verts_packed()
            obj_param[obj_id]['full_mesh'] = full_mesh
            obj_param[obj_id]['xyz_mesh'] = mesh_convert_coord(mesh.clone())
            obj_param[obj_id]['pts'] = pts.cpu().numpy()
            obj_param[obj_id]['diameter'] = info['diameter'] / scale
            # x_min, y_min, z_min = info['min_x'], info['min_y'], info['min_z']
            # x_max, y_max, z_max = x_min+info['size_x'], y_min+info['size_y'], y_min+info['size_y']
            # obj_param[obj_id]['bbox_3d'] = torch.tensor([
            #     [x_min, y_min, z_min], [x_max, y_min, z_min], 
            #     [x_max, y_max, z_min], [x_min, y_max, z_min],
            #     [x_min, y_min, z_max], [x_max, y_min, z_max], 
            #     [x_max, y_max, z_max], [x_min, y_max, z_max]]).to(torch.float) / scale
            obj_param[obj_id]['symmetry'] = ('symmetries_discrete' in info.keys() or 'symmetries_continuous' in info.keys())
            obj_param[obj_id]['scale'] = scale
        return obj_param

    def get_texture_mesh_with_trimesh(self, path):
        mesh = trimesh.load_mesh(path)
        vertices = torch.FloatTensor(mesh.vertices)
        faces = torch.Tensor(mesh.faces)
        faces = faces.type(torch.int32)
        uv_coordinates = torch.FloatTensor(mesh.visual.uv)
        verts_uvs = uv_coordinates[None,...]
        faces_uvs = faces[None,...]
        ply_model = load_ply(path)
        texture_path = f'{os.path.dirname(path)}/{ply_model["texture_file"]}'
        texture = transforms.ToTensor()(Image.open(texture_path))
        texture = torch.unsqueeze(texture, 0)
        texture = torch.permute(texture, (0, 2, 3, 1))
        tex = TexturesUV(maps=texture, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
        mesh = Meshes(verts=[vertices], faces=[faces], textures=tex)
        return mesh

    def preprocess_batch(self, que, obj):
        ref = defaultdict()
        device = que['RT'].device
        bsz, N_ref = que['K'].shape[:2]
        scale = torch.tensor(obj['scale']).view(-1, 1, 1).to(device)
        que['RT'][..., :3, 3] = que['RT'][..., :3, 3] / scale
        ref_RTs = []
        for i in range(bsz):
            ref_RTs.append(jitter_RT_with_scale(que['RT'][i], obj['scale'][i], jitter_angle_dis=(-15, 15), jitter_x_dis=(-15, 15), jitter_y_dis=(-15, 15), jitter_z_dis=(-50, 50)))
        ref['RT'] = torch.stack(ref_RTs)
        ref['K'] = que['K']
        ref['K_img'] = ref['K']
        # que['RT_coarse'][..., :3, 3] = que['RT_coarse'][..., :3, 3] / scale
        # if self.use_carved:
        #     source = obj['source'].expand(bsz, N_ref, 4, self.carving_size, self.carving_size, self.carving_size)
        # else:
        #     source = obj['source']
        #ref['source'] = obj['xyz_mesh']
        xyz_mesh = [x.to(device) for x in obj['xyz_mesh']]
        full_mesh = [x.to(device) for x in obj['full_mesh']]
        ref['source'] = xyz_mesh
        ref['xyz'] = render_geometry_scflow(xyz_mesh, ref['K'], 
            ref['RT'], self.image_size, self.geo_size, represent_mode = self.represent_mode)
        ref['xyz_origin'] = render_geometry(xyz_mesh, ref['K'], 
            ref['RT'], self.image_size, self.image_size, represent_mode = self.represent_mode)
        que['xyz'] = render_geometry_scflow(xyz_mesh, que['K'], 
			que['RT'], self.image_size, self.geo_size, represent_mode = self.represent_mode)
        que['xyz_origin'] = render_geometry(xyz_mesh, que['K'], 
            que['RT'], self.image_size, self.image_size, represent_mode = self.represent_mode)
        
        # ref['xyz'] = render_geometry(xyz_mesh, ref['K'], 
        #     ref['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
        # que['xyz'] = render_geometry(xyz_mesh, que['K'], 
		# 	que['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
        ref['image'] = render_geometry(
                            full_mesh, ref['K'], ref['RT'], self.image_size, self.image_size, represent_mode='rgb')
        ref['image'] = (ref['image'] + 1) / 2
        ref['mask'] = torch.round((ref['image'] > 0.0).to(torch.float).mean(dim=-3, keepdims=True))
        #image = (image[0] + 1) / 2
        
        # ref['xyz'] = render_geometry(source, ref['K_img'], 
		# 	ref['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
        # que['xyz'] = render_geometry(ref['source'], que['K'], 
		# 	que['RT'], self.image_size, self.image_size, self.N_z, self.N_freq, 'xyz')
        flow_gt = get_flow_from_delta_pose_and_xyz_scflow(que['RT'], que['K'], ref['xyz_origin'], ref['mask'], que['xyz'])
        #flow_gt = get_flow_from_delta_pose_and_xyz(que['RT'], que['K'], ref['xyz'], ref['mask'], que['xyz'])
        #from torchvision.utils import flow_to_image
        #a = flow_to_image(flow_gt.squeeze())
        #img = a[0].permute(1,2,3).cpu().numpy()
        #plt.imsave("ref.jpg", ref['image'][0][0].permute(1,2,0).cpu().numpy())
        return que, ref, flow_gt

    def load_sample(self, input_data, bbox=None, noisy=False):
        data = dict(list(deepcopy(input_data[0]).items()) + list(deepcopy(input_data[1]).items()))
        if bbox is None:
            bbox = bbox_add_noise(data['bbox_obj'], std_rate=0.1) if noisy else data['bbox_obj']
        bbox = squaring_boxes(bbox, lamb=1.1)
        image = np.asarray(Image.open(data['image']).convert("RGB")) if isinstance(data['image'], str) else data['image']
        mask = np.asarray(Image.open(data['mask'])) if isinstance(data['mask'], str) else data['mask']
        mask_visib = np.asarray(Image.open(data['mask_visib'])) if isinstance(data['mask_visib'], str) else data['mask_visib']
        image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)/255.0
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(1)/255.0
        mask_visib = torch.tensor(mask_visib).unsqueeze(0).unsqueeze(1)/255.0
        image = image_cropping(image, bbox, [self.image_size, self.image_size])
        mask = image_cropping(mask, bbox, [self.ref_size, self.ref_size]).round()
        mask_visib = image_cropping(mask_visib, bbox, [self.ref_size, self.ref_size]).round()
        K = get_K_crop_resize(data['K'], bbox, [self.image_size, self.image_size])
        K_ref = get_K_crop_resize(data['K'], bbox, [self.ref_size, self.ref_size])
        obj_index = self.obj_list.index(data['obj_id']) 
        sample = {
            'scene_id': data['scene_id'],
            'im_id': data['im_id'],
            'inst_id': data['inst_id'],
            'obj_id': data['obj_id'],
            'obj_index': obj_index,
            'image': image,
            'mask': mask,
            'mask_visib': mask_visib,
            'RT': data['RT'],
            'bbox': bbox,
            'K': K,
            'K_ref': K_ref}
        return sample

    def __getitem__(self, step):
        if self.is_train:
            real_or_pbr = np.random.randint(len(self.id_list))
            id_list = self.id_list[real_or_pbr]
            scene_id, im_id, inst_id = id_list[step % len(id_list)]
            image_data = self.dataset[real_or_pbr]['image_data'][(scene_id, im_id)]
            instance_data = self.dataset[real_or_pbr]['instance_data'][(scene_id, im_id)][inst_id]
            query = self.load_sample((image_data, instance_data), noisy=True)
        else:
            scene_id, im_id, inst_id = self.id_list[step]
            image_data = self.dataset['image_data'][(scene_id, im_id)]
            instance_data = self.dataset['instance_data'][(scene_id, im_id)][inst_id]
            bbox = self.test_bbox[(scene_id, im_id)][inst_id]
            query = self.load_sample((image_data, instance_data), bbox=bbox, noisy=False)
        return query

    def shuffle_data(self):
        for i, id_list in enumerate(self.id_list):
            random.shuffle(self.id_list[i])

    def __len__(self):
        return int(1e10) if self.is_train else len(self.id_list)



    def batch_object(self, batch_obj_id):
        obj = defaultdict()
        #batch_obj_id = batch_obj_id.tolist()
        obj['full_mesh'] = [self.object_param[id]['full_mesh'] for id in batch_obj_id]
        obj['xyz_mesh'] = [self.object_param[id]['xyz_mesh'] for id in batch_obj_id]
        obj['diameter'] = [self.object_param[id]['diameter'] for id in batch_obj_id]
        obj['symmetry'] = [self.object_param[id]['symmetry'] for id in batch_obj_id]
        obj['pts'] = [self.object_param[id]['pts'] for id in batch_obj_id]
        obj['scale'] = [self.object_param[id]['scale'] for id in batch_obj_id]
        return obj 

    def collate_fn(self, batch):
        ids = defaultdict()
        que = defaultdict()

        ids['scene_id'] = [q['scene_id'] for q in batch]
        ids['im_id'] = [q['im_id'] for q in batch]
        ids['inst_id'] = [q['inst_id'] for q in batch]
        ids['obj_id'] = [q['obj_id'] for q in batch]
        que['obj_index'] = torch.tensor([q['obj_index'] for q in batch])
        
        que['RT'] = torch.stack([q['RT'] for q in batch]).unsqueeze(1)
        que['K'] = torch.stack([q['K'] for q in batch]).unsqueeze(1)
        que['K_ref'] = torch.stack([q['K_ref'] for q in batch]).unsqueeze(1)
        que['image'] = torch.stack([q['image'] for q in batch])
        que['mask'] = torch.stack([q['mask'] for q in batch])
        que['mask_visib'] = torch.stack([q['mask_visib'] for q in batch])
        return ids, que
    
class lmo_dataset(BOP_Dataset):
    def __init__(self, cfg, is_train=True, is_memory=False):
        super(lmo_dataset, self).__init__(cfg, is_train, is_memory)
        self.root_path = f'Dataset/BOP_SPECIFIC/{cfg.dataset_name}'
        self.ref_path = f'{self.root_path}/ref'
        self.obj_info = load_json_fast(f'{self.root_path}/models_eval/models_info.json')
        self.object_param = self.set_object_scflow(self.obj_list, include_textures=True)
        print(f'setting train dataset...')
        self.dataset = []
        self.id_list = []
        if 'real' in self.data_domain:
            print(f'setting real dataset...')
            train_ids = self.load_train_ids()
            train_dataset = self.load_dataset(train_ids, 'train')
            self.dataset += [train_dataset]
            real_id_list = []
            for (scene_id, im_id), instances in train_dataset['instance_data'].items():
                for inst_id, _ in instances.items():
                    real_id_list.append([scene_id, im_id, inst_id])
            random.shuffle(real_id_list)
            self.id_list += [real_id_list]

        if 'pbr' in self.data_domain:
            print(f'setting pbr dataset...')
            pbr_id_list = []
            pbr_ids = self.load_pbr_ids()
            pbr_dataset = self.load_dataset(pbr_ids, 'train_pbr')
            self.dataset += [pbr_dataset]
            for (scene_id, im_id), instances in pbr_dataset['instance_data'].items():
                for inst_id, _ in instances.items():
                    pbr_id_list.append([scene_id, im_id, inst_id])
            random.shuffle(pbr_id_list)
            self.id_list += [pbr_id_list]
       

    def load_train_ids(self):
        scene_img_file_list = glob(f'{self.root_path}/index/*train*.txt')
        scene_im_dict = defaultdict(list)
        for scene_img_file in scene_img_file_list:
            with open(scene_img_file) as f:
                scene_img_list = f.read().splitlines()
            for scene_img in scene_img_list:
                scene_id, im_id = scene_img.split('/')
                scene_id, im_id = int(scene_id), int(im_id)-1
                scene_im_dict[scene_id] += [im_id]
        ids = defaultdict(lambda: defaultdict(list))
        for scene_id, im_list in scene_im_dict.items():
            scene_gt = load_scene_gt(f'{self.root_path}/train/{scene_id:06}/scene_gt.json')
            for im_id in im_list:
                for inst_id in range(len(scene_gt[im_id])):
                    ids[scene_id][im_id] += [inst_id]
        return ids

    def load_test_ids(self):
        scene_img_file = f'{self.root_path}/index/test.txt'
        scene_im_dict = defaultdict(list)
        with open(scene_img_file) as f:
            scene_img_list = f.read().splitlines()
        for scene_img in scene_img_list:
            scene_id, im_id = scene_img.split('/')
            scene_id, im_id = int(scene_id), int(im_id)-1
            scene_im_dict[scene_id] += [im_id]
        ids = defaultdict(lambda: defaultdict(list))
        inst_id_of_obj_id = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        for scene_id, im_list in scene_im_dict.items():
            scene_gt = load_scene_gt(f'{self.root_path}/test/{scene_id:06}/scene_gt.json')
            for im_id in im_list:
                for inst_id in range(len(scene_gt[im_id])):
                    ids[scene_id][im_id] += [inst_id]
                    obj_id = scene_gt[im_id][inst_id]['obj_id']
                    inst_id_of_obj_id[scene_id][im_id][obj_id] = inst_id
        return ids, inst_id_of_obj_id


# class lmo_dataset(BOP_Dataset):
#     def __init__(self, cfg, is_train=True, is_memory=False):
#         super(lmo_dataset, self).__init__(cfg, is_train, is_memory)
#         self.root_path = f'Dataset/BOP_SPECIFIC/{cfg.dataset_name}'
#         if self.represent_mode == 'rgb':
#             self.object_param = set_object(f'{self.root_path}/models', self.obj_list, include_textures=True)
#         else:
#             self.object_param = set_object(f'{self.root_path}/models_eval', self.obj_list)
#         print(f'setting train dataset...')
#         train_ids = self.load_train_ids()
#         train_dataset = self.load_dataset(train_ids, 'train')
#         ref_data = self.select_reference(train_dataset)
#         self.reference = {k: self.load_reference(k, v, with_coord=True) for k, v in ref_data.items()}
#         if self.is_train:
#             self.dataset = [train_dataset]
#             real_id_list = []
#             for (scene_id, im_id), instances in train_dataset['instance_data'].items():
#                 for inst_id, _ in instances.items():
#                     real_id_list.append([scene_id, im_id, inst_id])
#             random.shuffle(real_id_list)
#             self.id_list = [real_id_list]
#             if self.is_pbr:
#                 pbr_id_list = []
#                 print(f'setting pbr dataset...')
#                 pbr_ids = self.load_pbr_ids()
#                 pbr_dataset = self.load_dataset(pbr_ids, 'train_pbr')
#                 self.dataset += [pbr_dataset]
#                 for (scene_id, im_id), instances in pbr_dataset['instance_data'].items():
#                     for inst_id, _ in instances.items():
#                         pbr_id_list.append([scene_id, im_id, inst_id])
#                 random.shuffle(pbr_id_list)
#                 self.id_list += [pbr_id_list]
#         else:
#             print(f'setting test dataset...')
#             test_ids, inst_id_of_obj_id = self.load_test_ids()
#             test_dataset = self.load_dataset(test_ids, 'test')
#             self.dataset = test_dataset
#             self.test_bbox = self.load_test_bbox(inst_id_of_obj_id)
#             self.id_list = []
#             for (scene_id, im_id), instances in test_dataset['instance_data'].items():
#                 for inst_id, _ in instances.items():
#                     self.id_list.append([scene_id, im_id, inst_id])

#     def load_train_ids(self):
#         scene_img_file_list = glob(f'{self.root_path}/index/*train*.txt')
#         scene_im_dict = defaultdict(list)
#         for scene_img_file in scene_img_file_list:
#             with open(scene_img_file) as f:
#                 scene_img_list = f.read().splitlines()
#             for scene_img in scene_img_list:
#                 scene_id, im_id = scene_img.split('/')
#                 scene_id, im_id = int(scene_id), int(im_id)-1
#                 scene_im_dict[scene_id] += [im_id]
#         ids = defaultdict(lambda: defaultdict(list))
#         for scene_id, im_list in scene_im_dict.items():
#             scene_gt = load_scene_gt(f'{self.root_path}/train/{scene_id:06}/scene_gt.json')
#             for im_id in im_list:
#                 for inst_id in range(len(scene_gt[im_id])):
#                     ids[scene_id][im_id] += [inst_id]
#         return ids

#     def load_test_ids(self):
#         scene_img_file = f'{self.root_path}/index/test.txt'
#         scene_im_dict = defaultdict(list)
#         with open(scene_img_file) as f:
#             scene_img_list = f.read().splitlines()
#         for scene_img in scene_img_list:
#             scene_id, im_id = scene_img.split('/')
#             scene_id, im_id = int(scene_id), int(im_id)-1
#             scene_im_dict[scene_id] += [im_id]
#         ids = defaultdict(lambda: defaultdict(list))
#         inst_id_of_obj_id = defaultdict(lambda: defaultdict(lambda: defaultdict()))
#         for scene_id, im_list in scene_im_dict.items():
#             scene_gt = load_scene_gt(f'{self.root_path}/test/{scene_id:06}/scene_gt.json')
#             for im_id in im_list:
#                 for inst_id in range(len(scene_gt[im_id])):
#                     ids[scene_id][im_id] += [inst_id]
#                     obj_id = scene_gt[im_id][inst_id]['obj_id']
#                     inst_id_of_obj_id[scene_id][im_id][obj_id] = inst_id
#         return ids, inst_id_of_obj_id

#     def load_test_bbox(self, inst_id_of_obj_id):
#         bbox_dict = defaultdict(lambda: defaultdict(dict))
#         test_bbox_path = f'{self.root_path}/test_bboxes/faster_R50_FPN_AugCosyAAE_HalfAnchor_lmo_pbr_lmo_fuse_real_all_8e_test_480x640.json'
#         bboxes = load_json(test_bbox_path)
#         for scene_im_id, instances in bboxes.items():
#             scene_id = int(scene_im_id.split('/')[0])
#             im_id = int(scene_im_id.split('/')[1])
#             max_instances = []
#             obj_list = list(inst_id_of_obj_id[scene_id][im_id].keys())
#             for obj_id in obj_list:
#                 obj = [inst for inst in instances if inst['obj_id'] == obj_id]
#                 obj_score = [inst['score'] for inst in instances if inst['obj_id'] == obj_id]
#                 inst_id = inst_id_of_obj_id[scene_id][im_id][obj_id]
#                 if len(obj) > 0:
#                     max_obj = obj[np.argmax(np.array(obj_score))]                 
#                 else:
#                     max_obj = {'bbox_est': [270, 190, 100, 100]}        # default bbox 
#                 max_instances.append((inst_id, max_obj))   
#             for inst_id, inst in max_instances:
#                 bbox = inst['bbox_est']
#                 bbox[2] += bbox[0]
#                 bbox[3] += bbox[1]
#                 bbox_dict[(scene_id, im_id)][inst_id] = torch.tensor(bbox)
#         return bbox_dict



class ycbv_dataset(BOP_Dataset):
    def __init__(self, cfg, is_train=True, is_memory=False):
        super(ycbv_dataset, self).__init__(cfg, is_train, is_memory)
        self.root_path = f'Dataset/BOP_SPECIFIC/{cfg.dataset_name}'
        self.ref_path = f'{self.root_path}/ref'
        self.obj_info = load_json_fast(f'{self.root_path}/models_eval/models_info.json')
        self.object_param = self.set_object_scflow(self.obj_list, include_textures=True)
        print(f'setting train dataset...')
        self.dataset = []
        self.id_list = []
        if 'real' in self.data_domain:
            print(f'setting real dataset...')
            train_ids = self.load_train_ids()
            train_dataset = self.load_dataset(train_ids, 'train_real')
            self.dataset += [train_dataset]
            real_id_list = []
            for (scene_id, im_id), instances in train_dataset['instance_data'].items():
                for inst_id, _ in instances.items():
                    real_id_list.append([scene_id, im_id, inst_id])
            random.shuffle(real_id_list)
            self.id_list += [real_id_list]

        if 'pbr' in self.data_domain:
            print(f'setting pbr dataset...')
            pbr_id_list = []
            pbr_ids = self.load_pbr_ids()
            pbr_dataset = self.load_dataset(pbr_ids, 'train_pbr')
            self.dataset += [pbr_dataset]
            for (scene_id, im_id), instances in pbr_dataset['instance_data'].items():
                for inst_id, _ in instances.items():
                    pbr_id_list.append([scene_id, im_id, inst_id])
            random.shuffle(pbr_id_list)
            self.id_list += [pbr_id_list]
       

    def load_train_ids(self):
        scene_img_file = f'{self.root_path}/index/train.txt'
        scene_im_dict = defaultdict(list)
        with open(scene_img_file) as f:
            scene_img_list = f.read().splitlines()
        for scene_img in scene_img_list:
            scene_id, im_id = scene_img.split('/')
            scene_id, im_id = int(scene_id), int(im_id)
            scene_im_dict[scene_id] += [im_id]
        ids = defaultdict(lambda: defaultdict(list))
        for scene_id, im_list in scene_im_dict.items():
            scene_gt = load_scene_gt(f'{self.root_path}/train_real/{scene_id:06}/scene_gt.json')
            for im_id in im_list:
                for inst_id in range(len(scene_gt[im_id])):
                    ids[scene_id][im_id] += [inst_id]
        return ids

    def load_test_ids(self):
        scene_img_file = f'{self.root_path}/index/test.txt'
        scene_im_dict = defaultdict(list)
        with open(scene_img_file) as f:
            scene_img_list = f.read().splitlines()
        for scene_img in scene_img_list:
            scene_id, im_id = scene_img.split('/')
            scene_id, im_id = int(scene_id), int(im_id)
            scene_im_dict[scene_id] += [im_id]
        ids = defaultdict(lambda: defaultdict(list))
        inst_id_of_obj_id = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        for scene_id, im_list in scene_im_dict.items():
            scene_gt = load_scene_gt(f'{self.root_path}/test/{scene_id:06}/scene_gt.json')
            for im_id in im_list:
                for inst_id in range(len(scene_gt[im_id])):
                    ids[scene_id][im_id] += [inst_id]
                    obj_id = scene_gt[im_id][inst_id]['obj_id']
                    inst_id_of_obj_id[scene_id][im_id][obj_id] = inst_id
        return ids, inst_id_of_obj_id
    
    # def load_test_ids_with_init(self):
    #     scene_img_file = f'{self.root_path}/index/test.txt'
    #     scene_im_dict = defaultdict(list)
    #     with open(scene_img_file) as f:
    #         scene_img_list = f.read().splitlines()
    #     for scene_img in scene_img_list:
    #         scene_id, im_id = scene_img.split('/')
    #         scene_id, im_id = int(scene_id), int(im_id)
    #         scene_im_dict[scene_id] += [im_id]
    #     ids = defaultdict(lambda: defaultdict(list))
    #     inst_id_of_obj_id = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    #     for scene_id, im_list in scene_im_dict.items():
    #         scene_gt = load_scene_gt(f'{self.root_path}/posecnn_init/{scene_id:06}/scene_gt.json')
    #         for im_id in im_list:
    #             for inst_id in range(len(scene_gt[im_id])):
    #                 ids[scene_id][im_id] += [inst_id]
    #                 obj_id = scene_gt[im_id][inst_id]['obj_id']
    #                 inst_id_of_obj_id[scene_id][im_id][obj_id] = inst_id
    #     return ids, inst_id_of_obj_id

    def load_test_bbox(self, inst_id_of_obj_id):
        bbox_dict = defaultdict(lambda: defaultdict(dict))
        test_bbox_path = f'{self.root_path}/test_bboxes/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json'
        bboxes = load_json(test_bbox_path)
        for scene_im_id, instances in bboxes.items():
            scene_id = int(scene_im_id.split('/')[0])
            im_id = int(scene_im_id.split('/')[1])
            max_instances = []
            obj_list = list(inst_id_of_obj_id[scene_id][im_id].keys())
            for obj_id in obj_list:
                obj = [inst for inst in instances if inst['obj_id'] == obj_id]
                obj_score = [inst['score'] for inst in instances if inst['obj_id'] == obj_id]
                inst_id = inst_id_of_obj_id[scene_id][im_id][obj_id]
                if len(obj) > 0:
                    max_obj = obj[np.argmax(np.array(obj_score))]                 
                else:
                    max_obj = {'bbox_est': [270, 190, 100, 100]}        # default bbox 
                max_instances.append((inst_id, max_obj))   
            for inst_id, inst in max_instances:
                bbox = inst['bbox_est']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bbox_dict[(scene_id, im_id)][inst_id] = torch.tensor(bbox)
        return bbox_dict

class OcclusionAugmentation(object):
    def __init__(self, p=0.5):
        self.p = p
        self.affine = transforms.RandomAffine((0, 180), (0.3, 0.3), (0.5, 0.8))
    def __call__(self, image, visib_mask, obj_id):
        bsz, n_view, c, h, w = image.shape
        mask_h, mask_w = visib_mask.shape[-2:]
        #obj_id = obj_id.tolist()
        v_mask = F.interpolate(visib_mask.flatten(0, 1), [h, w], mode='nearest').unflatten(0, (bsz, n_view))
        occlusion = torch.cat([image * v_mask, v_mask], dim=2)
        occlusion = self.affine(occlusion.flatten(0, 1)).unflatten(0, (bsz, n_view))
        occlusion_rgb, occlusion_mask = occlusion[:, :, :3], occlusion[:, :, 3:]
        inv_occlusion_mask = 1 - occlusion_mask
        for i, id in enumerate(obj_id):
            valid_idx = list(np.where(np.array(obj_id) != id)[0])
            if len(valid_idx) > 0 and self.p > np.random.uniform():
                occ_index = random.sample(valid_idx, 1)[0]
                image[i] = image[i] * inv_occlusion_mask[occ_index] + occlusion_rgb[occ_index] * occlusion_mask[occ_index]
                new_visib_mask = (v_mask[i] * inv_occlusion_mask[occ_index])
                visib_mask[i] = F.interpolate(new_visib_mask, [mask_h, mask_w], mode='nearest')
        return image, visib_mask

class ImageAugmentation(object):
    def __init__(self):
        color_aug =[transforms.ColorJitter([0.95, 1.05], [0.95, 1.05], [0.95, 1.05], [-0.05, 0.05]), 
                    transforms.GaussianBlur([5, 5], (0.05, 0.1))] #(0.05, 0.1), (0.5, 1.0)
        self.applier = transforms.RandomApply(transforms=color_aug, p=0.5)
    def __call__(self, image):
        bsz, n_view, c, h, w = image.shape
        image = self.applier(image.flatten(0, 1)).unflatten(0, (bsz, n_view))
        return image

