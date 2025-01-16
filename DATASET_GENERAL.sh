# Dataset Setup
# mkdir -p Dataset/BOP_GENERAL
cd Dataset/BOP_GENERAL

# # download object files
dataset=megapose-gso
# mkdir -p ${dataset}
cd ${dataset}
# wget https://app.gazebosim.org/assets/scripts/download_collection.py
# wget https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/gso_models.json
# wget https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/key_to_shard.json
# mkdir -p model
# cd model 
# python3 ../download_collection.py -o "GoogleResearch" -c "Scanned Objects by Google Research"
# cd ..
zip_folder="model"
for file in "$zip_folder"/*.zip; do
	if [ -f "$file" ]; then
		folder_name=$(basename "$file" .zip)
		folder_path="$zip_folder/$folder_name"
		texture_src="$zip_folder/$folder_name/materials/textures/texture.png"
		texture_tgt="$zip_folder/$folder_name/meshes/texture.png"
		mkdir -p "$folder_path"
		unzip "$file" -d "$folder_path"
		mv "$texture_src" "$texture_tgt"
		# rm "$file"
	fi
done


# 다운로드 함수
# download() {
#     SHARD_ID=$1
#     URL_PREFIX=$2
#     URL_SUFFIX=$3
#     URL="$URL_PREFIX$SHARD_ID$URL_SUFFIX"
#     FILENAME="shard-$SHARD_ID.tar"

#     # 파일 다운로드
#     echo "다운로드 중: $URL"
#     wget "$URL" -O "$FILENAME"
# }
# # SHARD_ID의 시작과 끝
# START_ID=0
# END_ID=1039
# data=data
# mkdir -p ${data}
# cd ${data}
# for ID in $(seq $START_ID $END_ID); do
#     SHARD_ID=$(printf "%06d" $ID)
#     download "$SHARD_ID" "https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-gso/train_pbr_web/shard-" ".tar"
# done
# cd ../..

# data=MegaPose_ShapeNetCore_Web/data
# mkdir -p ${data}
# cd ${data}
# for ID in $(seq $START_ID $END_ID); do
#     SHARD_ID=$(printf "%06d" $ID)
#     download "$SHARD_ID" "https://bop.felk.cvut.cz/media/data/bop_datasets/bop23_datasets/megapose-shapenet/train_pbr_web/shard-" ".tar"
# done