import json
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import ImageColor

local_machine = (platform.node() == 'dalcowks')

# project directory
project_folder = Path(__file__).resolve().parents[1]
split = 'test'
scene_folder       = project_folder / '..' / 'SphericalMask' / 'dataset' / 'scannetv2' / split
proxe_base_folder  = project_folder / 'proxe'
scene_cache_folder = proxe_base_folder / 'scene_segmentation'
sdf_folder         = proxe_base_folder / 'sdf'
# POSA files downloaded from: https://posa.is.tue.mpg.de/index.html
posa_folder        = proxe_base_folder / 'POSA_dir'
mesh_ds_folder     = posa_folder / 'mesh_ds'
# smplx models
smplx_model_folder = proxe_base_folder / 'models_smplx_v1_1' / 'models'
# mesh upsample and downsample weights
mesh_operation_file = project_folder / 'data' / 'mesh_operation.npz'
# tranformation matrix between PROX and POSA scenes
scene_registration_file = project_folder / 'data' / 'scene_registration.pkl'
# checkpoints
checkpoint_folder = project_folder / 'checkpoints'
checkpoint_folder.mkdir(parents=True, exist_ok=True)
# rendering and results
results_folder = project_folder / 'results'
results_folder.mkdir(parents=True, exist_ok=True)
render_folder = project_folder / 'render'
render_folder.mkdir(parents=True, exist_ok=True)

# scene names
scene_names = ["scene0", "scene1", "scene2"]

# sequence names
recordings_temporal = project_folder / 'configuration' / 'recordings_temporal.txt'
sequence_names      = [sequence.split('\n')[0] for sequence in recordings_temporal.open().readlines()]

# interaction names
atomic_interaction_names = ['sit on-chair', 'sit on-sofa', 'sit on-bed', 'sit on-cabinet', 'sit on-table', 'stand on-floor', 'stand on-table', 'stand on-bed', 'stand on-chest_of_drawers', 'lie on-sofa', 'lie on-bed', 'touch-table', 'touch-board_panel', 'touch-tv_monitor', 'touch-shelving', 'touch-wall', 'touch-shelving']
atomic_interaction_names_include_motion = ['jump on-sofa', 'step down-table', 'touch-shelving', 'sit down-sofa', 'step up-table', 'side walk-floor', 'turn-floor', 'sit down-chair', 'stand up-bed', 'step up-sofa', 'step down-sofa', 'step down-chair', 'touch-board_panel', 'sit on-seating', 'sit on-chair', 'walk on-floor', 'sit on-bed', 'stand on-table', 'stand up-sofa', 'turnover-floor', 'lie on-sofa', 'lie down-sofa', 'a pose-floor', 'touch-tv_monitor', 'stand up-chair', 'sit up-sofa', 'restfoot-chair', 'stand on-bed', 'step back-floor', 'touch-chair', 'step up-chair', 'move leg-sofa', 'move on-sofa', 'touch-chest_of_drawers', 'touch-sofa', 'stand up-cabinet', 'sit on-stool', 'lie on-bed', 'touch-table', 'lie on-seating', 'touch-wall', 'stand on-floor', 'sit on-sofa', 'move leg-bed', 'sit on-table', 'sit on-cabinet', 'restfoot-stool', 'sit down-cabinet', 'stand on-chest_of_drawers', 'sit down-bed']
atomic_interaction_names_include_motion_train = ['sit on-sofa', 'touch-shelving', 'touch-tv_monitor', 'sit down-sofa', 'jump on-sofa', 'touch-chair', 'step down-chair', 'walk on-floor', 'touch-chest_of_drawers', 'sit down-bed', 'sit on-table', 'move on-sofa', 'stand on-chest_of_drawers', 'turn-floor', 'lie on-sofa', 'stand up-bed', 'lie on-bed', 'step up-sofa', 'side walk-floor', 'sit down-cabinet', 'stand up-chair', 'stand up-cabinet', 'touch-sofa', 'sit on-cabinet', 'a pose-floor', 'move leg-sofa', 'sit on-bed', 'touch-wall', 'sit on-chair', 'step down-table', 'stand up-sofa', 'sit up-sofa', 'touch-table', 'step up-chair', 'stand on-table', 'step down-sofa', 'sit down-chair', 'stand on-floor', 'stand on-bed', 'touch-board_panel', 'lie down-sofa', 'step up-table']
composed_interaction_names = ['sit on-chair+touch-table', 'sit on-sofa+touch-table', 'stand on-floor+touch-board_panel', 'stand on-floor+touch-table', 'stand on-floor+touch-tv_monitor', 'stand on-floor+touch-shelving', 'stand on-floor+touch-wall']
test_composed_interaction_names = ['sit on-chair+touch-table', 'stand on-floor+touch-board_panel', 'stand on-floor+touch-table']
interaction_names = atomic_interaction_names_include_motion_train + composed_interaction_names

# load category name and visualization color (scannetv2 classes)
scannet_benchmark_labels = [
    'void',             # 0
    'wall',             # 1
    'floor',            # 2
    'cabinet',          # 3
    'bed',              # 4
    'chair',            # 5
    'sofa',             # 6
    'table',            # 7
    'door',             # 8
    'window',           # 9
    'bookshelf',        # 10
    'picture',          # 11
    'counter',          # 12
    'void',             # 13
    'desk',             # 14
    'void',             # 15
    'curtain',          # 16
    'void',             # 17
    'void',             # 18
    'void',             # 19
    'void',             # 20
    'void',             # 21
    'void',             # 22
    'void',             # 23
    'refrigerator',     # 24
    'void',             # 25
    'void',             # 26
    'void',             # 27
    'shower curtain',   # 28
    'void',             # 29
    'void',             # 30
    'void',             # 31
    'void',             # 32
    'toilet',           # 33
    'sink',             # 34
    'void',             # 35
    'bathtub',          # 36
    'void',             # 37
    'void',             # 38
    'otherfurniture',   # 39
    'void'              # 40
]
np.random.seed(42)
colors = (np.random.rand(len(scannet_benchmark_labels), 3) * 255).astype(int).tolist()
data = {
    'mpcat40': scannet_benchmark_labels,
    'color': colors
}
category_dict = pd.DataFrame(data)
obj_category_num = len(scannet_benchmark_labels)

# human body param
num_pca_comps = 6
smplx_param_names = ['betas', 'global_orient', 'transl', 'body_pose', 'left_hand_pose', 'right_hand_pose']
smplx_param_names += ['jaw_pose', 'leye_pose', 'reye_pose', 'expression']
used_smplx_param_names = ['transl', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'betas']  # these are used in diversity evaluation

# body part segmentation
body_parts = ['back', 'gluteus', 'L_Hand', 'R_Hand', 'L_Leg', 'R_Leg', 'thighs']
body_part_vertices = {}
for body_part in body_parts:
    with open(proxe_base_folder / 'body_segments' / (body_part + '.json'), 'r') as file:
        body_part_vertices[body_part] = json.load(file)['verts_ind']
#https://github.com/Meshcapade/wiki/blob/main/assets/SMPL_body_segmentation/smplx/smplx_vert_segmentation.json
with open(project_folder / 'configuration' / 'smplx_vert_segmentation.json', 'r') as file:
    body_part_vertices_full = json.load(file)
upper_body_parts = ["rightHand", "leftArm", "rightArm", "leftHandIndex1", "rightHandIndex1", "leftForeArm", "rightForeArm", "leftHand"]
lower_body_parts = ["rightUpLeg", "leftLeg", "leftToeBase", "leftFoot", "rightFoot", "rightLeg", "rightToeBase", "leftUpLeg"]

# map action to corresponding body parts
action_names = ['sit on', 'lie on', 'stand on', 'touch', 'step back', 'restfoot', 'step down', 'turn', 'jump on', 'sit up', 'stand up', 'turnover', 'sit down', 'move on', 'lie down', 'move leg', 'walk on', 'a pose', 'step up', 'side walk']
action_names_train = ['sit on', 'lie on', 'stand on', 'touch', 'jump on', 'turn', 'move leg', 'stand up', 'sit down', 'sit up', 'side walk', 'step down', 'walk on', 'a pose', 'lie down', 'step up', 'move on']
num_verb = len(action_names)
num_noun = 42
maximum_atomics = 2
action_body_part_mapping = {
    'sit on': ['gluteus', 'thighs'],
    'lie on': ['back', 'gluteus', 'thighs'],
    'stand on': ['L_Leg', 'R_Leg'],
    'touch': ['L_Hand', 'R_Hand'],
}
