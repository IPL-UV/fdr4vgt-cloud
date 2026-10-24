# Pipeline environment setup

The release pipeline uses a dedicated virtual environment. Do not reuse
`release/venv` (it was created with `--system-site-packages`).

```bash
uv venv --python 3.12 /data/users/julio/experiments/cloud-proba-spot/release/.venv-release
source /data/users/julio/experiments/cloud-proba-spot/release/.venv-release/bin/activate
uv pip install \
  tacotoolbox[docs]==0.26.9 \
  tacoreader==2.4.21 \
  mlstac==0.4.9 \
  rasterio \
  rio-cogeo==7.0.4 \
  torch==2.7.0 \
  segmentation-models-pytorch==0.5.0 \
  safetensors \
  pyarrow \
  huggingface_hub \
  jupyter \
  nbconvert \
  pytest \
  ruff \
  scikit-learn \
  einops
uv pip freeze > requirements-release.txt
```

For the torch 2.14.1 compatibility check:

```bash
uv venv --python 3.12 /data/users/julio/experiments/cloud-proba-spot/release/.venv-torch214
source /data/users/julio/experiments/cloud-proba-spot/release/.venv-torch214/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install mlstac==0.4.9 rasterio
```

All paths inside the pipeline are absolute and point at local data under
`/data`. The released repository never contains these paths.
