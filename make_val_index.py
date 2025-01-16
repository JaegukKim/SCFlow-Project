import random

# 파일 읽기
with open('Dataset/BOP_SPECIFIC/ycbv/index/test.txt', 'r') as file:
    lines = file.readlines()

# scene_id별로 img_id를 저장할 딕셔너리
scene_dict = {}

# 입력된 데이터를 딕셔너리로 분류
for line in lines:
    scene_id, img_id = line.strip().split('/')
    if scene_id not in scene_dict:
        scene_dict[scene_id] = []
    scene_dict[scene_id].append(img_id)

# 새로운 파일에 선택된 img_id를 저장
with open('Dataset/BOP_SPECIFIC/ycbv/index/val.txt', 'w') as file:
    for scene_id in sorted(scene_dict.keys(), key=int):
        # 각 scene에서 50개의 img_id를 무작위로 선택
        selected_img_ids = random.sample(scene_dict[scene_id], min(50, len(scene_dict[scene_id])))
        for img_id in selected_img_ids:
            file.write(f"{scene_id}/{img_id}\n")