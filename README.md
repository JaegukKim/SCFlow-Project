# SCFlow Project

## Training
- Train:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/train_scflow_base_ref.py --config-path --configs/scflow_ycbv.yaml

## Inference
- Test:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/infoer_scflow_base_posecnn.py --config-path --saved/bop_specific/scflow_base_ref/scflow_ycbv.yaml
