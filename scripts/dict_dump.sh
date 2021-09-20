#!/bin/bash

source /data/vision/polina/scratch/dmoyer/bash_source.sh

PYTHONPATH="../shmpy/shmpy/:src/:${PYTHONPATH}" python src/zip_dump.py \
  --subj-idx ${SLURM_ARRAY_TASK_ID} \
  --hcp-table ${HCP_DATA_TABLE_LOC} \
  --hcp-zip-path ${HCP_ZIP_PATH}



