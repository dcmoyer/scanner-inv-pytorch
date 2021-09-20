

import pandas as pd
import numpy as np

#import SH as SHF
import shmpy
from dipy.io import read_bvals_bvecs

import torch
import nibabel as nib

import bvec_bval_tools as bbt
import joblib

from scipy import ndimage as ndi

import os


def work_volume( scan, mask, bvals, bvecs, target_bvals, bval_count_max, bval_thold=100, num_sh=8, mean_b0s=False):

    scan = scan[np.newaxis,...]
    scan = np.transpose(scan, axes=(0,4,1,2,3))

    mask = mask[np.newaxis,...]
    mask = np.transpose(mask, axes=(0,4,1,2,3))

    scan, bvals, bvecs = bbt.mean_b0s( bvals, bvecs, scan[0], bval_thold)
    scan = scan[np.newaxis,...]

    dilated_mask = ndi.morphology.binary_dilation(mask,iterations=1)

    output = {}
    output["orig_vol"] = scan
    output["mask"] = mask
    output["dilated_mask"] = dilated_mask

    sh_mat, reduced_weight_array, orig_sh_mat = bbt.b_transform(
        bvals, bvecs, target_bvals,
        bval_thold, num_sh,
        debug=False, pad=True, bval_count_max=bval_count_max
    )

    output["sh_mat"] = sh_mat
    output["reduced_weight_array"] = reduced_weight_array
    output["orig_sh_mat"] = orig_sh_mat 

    padded_vol, padded_bvals, padded_bvecs = bbt.reorg_volume(
        bvals, bvecs, target_bvals,
        bval_thold, num_sh,
        scan, pad=True, bval_count_max=bval_count_max
    )

    output["padded_vol"] = padded_vol
    output["padded_bvecs"] = padded_bvals
    output["padded_bvals"] = padded_bvecs

    wrapper = shmpy.Py_SH_Wrapper()
    num_coefs = wrapper.check_num_even_coef(num_sh)

    x,y,z = np.where(dilated_mask[0,0,...] > 0)

    N_B0s = np.sum(bvals < bval_thold)
    output["harmonic_volume"] = np.zeros( \
        [1,num_coefs * (len(target_bvals)-1) + N_B0s] + list(scan.shape[2:])\
    ) 

    batch_size = 1000

    #from tqdm import tqdm
    #for batch in tqdm(range(0,x.shape[0],batch_size)):
    for batch_idx,batch in enumerate(range(0,x.shape[0],batch_size)):

        print(batch_idx)
        x_batch = x[batch:min(batch+batch_size,x.shape[0])]
        y_batch = y[batch:min(batch+batch_size,x.shape[0])]
        z_batch = z[batch:min(batch+batch_size,x.shape[0])]
        batch_vol = scan[0][:,x_batch, y_batch, z_batch]

        projected_batch = bbt.project_volume( 
            bvals, bvecs, target_bvals,
            bval_thold, num_sh,
            batch_vol,
            wrapper=wrapper
        )

        output["harmonic_volume"][0][:,x_batch,y_batch,z_batch] = projected_batch

    return output


def get_dlists(hcp_table_path):
    df = pd.read_csv(hcp_table_path, sep="\t", header=0, comment="#")
    has_3T = []
    has_7T = []
    for idx, row in df.iterrows():
        subj_id = row["subj_id"]

        name = 1200
        scan_file = row[f"path_{name}"]
        mask_file = row["path_mask"]
        bval_file = row[f"path_{name}_bvals"]
        bvec_file = row[f"path_{name}_bvecs"]

        if os.path.exists(scan_file) and \
            os.path.exists(mask_file) and \
            os.path.exists(bval_file) and \
            os.path.exists(bvec_file): \
            has_3T.append(subj_id)

        name = "7T"
        scan_file = row[f"path_{name}"]
        mask_file = row["path_mask"]
        bval_file = row[f"path_{name}_bvals"]
        bvec_file = row[f"path_{name}_bvecs"]

        if os.path.exists(scan_file) and \
            os.path.exists(mask_file) and \
            os.path.exists(bval_file) and \
            os.path.exists(bvec_file): \
            has_7T.append(subj_id)

    has_both = list(set(has_3T).intersection(set(has_7T)))

    #TODO: undo this...but for ease of use...
    training = has_both[:10]
    validation = has_both[10:20]
    testing = has_both[20:30]
    return training, validation, testing

def dump_dicts(train, validation, testing, hcp_table_path, hcp_zip_path, target_idx=None):
    df = pd.read_csv(hcp_table_path, sep="\t", header=0,comment='#')
    df = df[df["subj_id"].isin(train+validation+testing)]
    df.reset_index(drop=True,inplace=True)
    print(df)

    scans_and_masks = []
    for idx, row in df.iterrows():

        if target_idx is not None and idx != target_idx:
            continue

        name = "1200"
        subj_id = row["subj_id"]
        print(subj_id)
        scan_file = row[f"path_{name}"]
        mask_file = row["path_mask"]
        bval_file = row[f"path_{name}_bvals"]
        bvec_file = row[f"path_{name}_bvecs"]

        #scan = nib.load(scan_file).header.get_data_shape()
        #mask = nib.load(mask_file).header.get_data_shape()
        #print(f"scan shape: {scan}")
        #print(f"mask shape: {mask}")        

        scan = nib.load(scan_file).get_fdata() / 300
        mask = nib.load(mask_file).get_fdata()
        print(f"scan shape: {scan.shape}")
        print(f"mask shape: {mask.shape}")
        if len(mask.shape) == 3:
            mask = mask[...,np.newaxis]

        bvals, bvecs = read_bvals_bvecs( bval_file, bvec_file )

        target_bvals = [0, 1000]
        bval_count_max = [1, 90]

        item_dict = work_volume( scan, mask, bvals, bvecs, target_bvals=target_bvals, bval_count_max=bval_count_max, mean_b0s=True)

        joblib.dump(item_dict,f"{hcp_zip_path}/{idx}_{name}.gz", ("gzip",3))

        name = "7T"
        subj_id = row["subj_id"]
        scan_file = row[f"path_{name}"]
        mask_file = row["path_mask"]
        bval_file = row[f"path_{name}_bvals"]
        bvec_file = row[f"path_{name}_bvecs"]

        #scan = nib.load(scan_file).header.get_data_shape()
        #mask = nib.load(mask_file).header.get_data_shape()
        #print(f"scan shape: {scan}")
        #print(f"mask shape: {mask}")        

        scan = nib.load(scan_file).get_fdata() / 300
        mask = nib.load(mask_file).get_fdata()
        print(f"scan shape: {scan.shape}")
        print(f"mask shape: {mask.shape}")
        if len(mask.shape) == 3:
            mask = mask[...,np.newaxis]

        bvals, bvecs = read_bvals_bvecs( bval_file, bvec_file )

        target_bvals = [0, 1000]
        #bval_count_max = [1, 64]
        #even though the 7T volumes have only 64 directions, we need 90 for padding
        bval_count_max = [1, 90] 

        item_dict = work_volume( scan, mask, bvals, bvecs, target_bvals=target_bvals, bval_count_max=bval_count_max, mean_b0s=True)

        joblib.dump(item_dict,f"{hcp_zip_path}/{idx}_{name}.gz", ("gzip",3))


if __name__ == "__main__":
    print("WARNING: use the env variable OMP_NUM_THREADS=<int> to control parallelism.")

    import argparse

    parser = argparse.ArgumentParser(description=\
        "Test Function to Create")

    parser.add_argument("--subj-idx", default=None, type=int)
    parser.add_argument("--hcp-table")
    parser.add_argument("--hcp-zip-path")

    hcp_table_path = args.hcp_table #"data/full_table_qc.tsv"
    args = parser.parse_args(hcp_table_path)

    print(f"subj_idx: {args.subj_idx}")
    train, val, test = get_dlists(hcp_table_path)

    dump_dicts(train, val, test, hcp_table_path, hcp_zip_path, target_idx=args.subj_idx)

