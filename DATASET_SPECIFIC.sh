# Dataset Setup

### Setting path
SRC=https://bop.felk.cvut.cz/media/data/bop_datasets
SRC_EXTRA=https://bop.felk.cvut.cz/media/data/bop_datasets_extra
mkdir -p Dataset/BOP_SPECIFIC
cd Dataset/BOP_SPECIFIC

### bop_core_bboxes
wget $SRC_EXTRA/bop23_default_detections_for_task4.zip --no-check-certificate               # test bounding box
unzip bop23_default_detections_for_task4.zip
rm bop23_default_detections_for_task4.zip

### YCBV Dataset
data=ycbv
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate                                                            # models
wget $SRC/${data}_test_bop19.zip --no-check-certificate                                                             # test
gdown https://drive.google.com/uc?id=1D42JfPa-6Y7pXPvu41faFwZkD8QoB5zd  --no-check-certificate                 # test_bbox, index
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_bop19.zip -d ${data}
unzip ${data}_bbox_index.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_bop19.zip
rm ${data}_bbox_index.zip

### LINEMOD Dataset
data=lm
wget $SRC/${data}_base.zip --no-check-certificate                                        
wget $SRC/${data}_models.zip --no-check-certificate                                                     # models
wget $SRC/${data}_test_all.zip --no-check-certificate                                                      # test
gdown https://drive.google.com/uc?id=1S1N3ZPvCfXPICCJJ1moXzhWA_4H2cMAP --no-check-certificate           # test_bbox, index
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_all.zip -d ${data}
unzip ${data}_bbox_index.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_all.zip
rm ${data}_bbox_index.zip
cd ${data}
ln -s test train
cd ..

### OCCLUSION dataset
data=lmo
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate                                                 # models
wget $SRC/${data}_test_all.zip --no-check-certificate                                                  # test
gdown https://drive.google.com/uc?id=1xzUc4hmFUgyJGWVEuedQ0nq9dUaUsKiW  --no-check-certificate      # test_bbox, index
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_all.zip -d ${data}
unzip ${data}_bbox_index.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_all.zip
rm ${data}_bbox_index.zip
cd ${data}
ln -s ../lm/test train                                                                             # shares training set with LM
cd ..

# ### other bop cores
data=tless
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate
wget $SRC/${data}_test_primesense_bop19.zip --no-check-certificate
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_primesense_bop19.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_primesense_bop19.zip

data=tudl
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate
wget $SRC/${data}_test_bop19.zip --no-check-certificate
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_bop19.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_bop19.zip

data=icbin
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate
wget $SRC/${data}_test_bop19.zip --no-check-certificate
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_bop19.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_bop19.zip

data=itodd
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate
wget $SRC/${data}_test_bop19.zip --no-check-certificate
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_bop19.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_bop19.zip

data=hb
wget $SRC/${data}_base.zip --no-check-certificate
wget $SRC/${data}_models.zip --no-check-certificate
wget $SRC/${data}_test_primesense_bop19.zip --no-check-certificate
unzip ${data}_base.zip
unzip ${data}_models.zip -d ${data}
unzip ${data}_test_primesense_bop19.zip -d ${data}
rm ${data}_base.zip
rm ${data}_models.zip
rm ${data}_test_primesense_bop19.zip
