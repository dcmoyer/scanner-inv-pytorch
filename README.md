
Inv Rep (HCP dMRI)
====

Usage (running from repo root directory):
```bash

PYTHONPATH="src/:${PYTHONPATH}" python \
  src/runner.py \
    --hcp-zip-path PATH_TO_OUTPUT_OF_DICTDUMP \
    --save-path params/

```

In order to run the data prep script `scripts/dict_dump.sh`, which in turn runs `src/zip_dump.py`, we need a CSV with the following columns:
```
index
subj_id
path_1200
path_mask
path_7T
path_1200_bvals
path_1200_bvecs
path_7T_bvals
path_7T_bvecs
```




