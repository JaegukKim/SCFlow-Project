# Dataset Setup

### Setting path
SRC=https://bop.felk.cvut.cz/media/data/bop_datasets
SRC_EXTRA=https://bop.felk.cvut.cz/media/data/bop_datasets_extra
mkdir -p Dataset/BOP_SPECIFIC
cd Dataset/BOP_SPECIFIC

### bop_core_bboxes
wget $SRC_EXTRA/bop23_default_detections_for_task1.zip --no-check-certificate               # test bounding box
unzip bop23_default_detections_for_task1.zip
rm bop23_default_detections_for_task1.zip

### bop_core_bboxes
wget $SRC_EXTRA/bop23_default_detections_for_task4.zip --no-check-certificate               # test bounding box
unzip bop23_default_detections_for_task4.zip
rm bop23_default_detections_for_task4.zip

# ### LINEMOD Dataset
# data=lm
# wget $SRC/${data}_base.zip --no-check-certificate                                        
# wget $SRC/${data}_models.zip --no-check-certificate                                                     # models
# wget $SRC/${data}_test_all.zip --no-check-certificate                                                   # test, train
# wget $SRC/${data}_train_pbr.zip --no-check-certificate                                                  # train
# gdown https://drive.google.com/uc?id=1S1N3ZPvCfXPICCJJ1moXzhWA_4H2cMAP --no-check-certificate           # test_bbox, index
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_test_all.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_bbox_index.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_test_all.zip
# rm ${data}_train_pbr.zip
# rm ${data}_bbox_index.zip
# cd ${data}
# ln -s test train
# cd ..

# ### OCCLUSION dataset
# data=lmo
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate                                                 # models
# wget $SRC/${data}_test_all.zip --no-check-certificate                                               # test, train
# gdown https://drive.google.com/uc?id=1xzUc4hmFUgyJGWVEuedQ0nq9dUaUsKiW  --no-check-certificate      # test_bbox, index
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_test_all.zip -d ${data}
# unzip ${data}_bbox_index.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_test_all.zip
# rm ${data}_bbox_index.zip
# cd ${data}
# ln -s ../lm/test train                                                                             # shares training set with LM
# ln -s ../lm/train_pbr train_pbr                                                                    # shares pbr set with LM
# cd ..

### YCBV Dataset
data=ycbv
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate                                                            # models
wget $SRC/${data}_test_bop19.zip --no-check-certificate                                                          # test
wget $SRC/${data}_train_pbr.zip --no-check-certificate                                                         # pbr
wget $SRC/${data}_train_real.zip --no-check-certificate                                                        # train
gdown https://drive.google.com/uc?id=1D42JfPa-6Y7pXPvu41faFwZkD8QoB5zd  --no-check-certificate                 # test_bbox, index
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_bop19.zip -d ${data}
unzip ${data}_train_pbr.zip -d ${data}
unzip ${data}_train_real.zip -d ${data}
unzip ${data}_bbox_index.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_bop19.zip
rm ${data}_train_pbr.zip
rm ${data}_train_real.zip
rm ${data}_bbox_index.zip

# ### other bop cores
# data=tless
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate
# wget $SRC/${data}_train_pbr.zip --no-check-certificate
# wget $SRC/${data}_train_primesense.zip --no-check-certificate
# wget $SRC/${data}_test_primesense_bop19.zip --no-check-certificate
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_train_primesense.zip -d ${data}
# unzip ${data}_test_primesense_bop19.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_train_pbr.zip
# rm ${data}_train_primesense.zip
# rm ${data}_test_primesense_bop19.zip

# data=tudl
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate
# wget $SRC/${data}_train_pbr.zip --no-check-certificate
# wget $SRC/${data}_train_real.zip --no-check-certificate
# wget $SRC/${data}_test_bop19.zip --no-check-certificate
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_train_real.zip -d ${data}
# unzip ${data}_test_bop19.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_train_pbr.zip
# rm ${data}_train_real.zip
# rm ${data}_test_bop19.zip

# data=icbin
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate
# wget $SRC/${data}_train_pbr.zip --no-check-certificate
# wget $SRC/${data}_train.zip --no-check-certificate
# wget $SRC/${data}_test_bop19.zip --no-check-certificate
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_train.zip -d ${data}
# unzip ${data}_test_bop19.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_train_pbr.zip
# rm ${data}_train.zip
# rm ${data}_test_bop19.zip

# data=itodd
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate
# wget $SRC/${data}_train_pbr.zip --no-check-certificate
# wget $SRC/${data}_val.zip --no-check-certificate
# wget $SRC/${data}_test_bop19.zip --no-check-certificate
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_val.zip -d ${data}
# unzip ${data}_test_bop19.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_train_pbr.zip
# rm ${data}_val.zip
# rm ${data}_test_bop19.zip

# data=hb
# wget $SRC/${data}_base.zip --no-check-certificate
# wget $SRC/${data}_models.zip --no-check-certificate
# wget $SRC/${data}_train_pbr.zip --no-check-certificate
# wget $SRC/${data}_val_primesense.zip --no-check-certificate
# wget $SRC/${data}_test_primesense_bop19.zip --no-check-certificate
# unzip ${data}_base.zip
# unzip ${data}_models.zip -d ${data}
# unzip ${data}_train_pbr.zip -d ${data}
# unzip ${data}_val_primesense.zip -d ${data}
# unzip ${data}_test_primesense_bop19.zip -d ${data}
# rm ${data}_base.zip
# rm ${data}_models.zip
# rm ${data}_train_pbr.zip
# rm ${data}_val_primesense.zip
# rm ${data}_test_primesense_bop19.zip