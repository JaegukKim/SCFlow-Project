# SCFlow Project

## Training
This code has been tested on a **Ubuntu 18.04** server with **CUDA 11.3**.

- Train:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scflow_base_ref.py --configs/scflow_ycbv.yaml
