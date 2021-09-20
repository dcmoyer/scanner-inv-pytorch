
import pandas as pd
import numpy as np

#import SH as SHF
import shmpy

import sklearn.preprocessing as skpp
from scipy.linalg import block_diag


##
## TODO: deal with unequal number of b0s
##

##
##
##

def sh_mat_helper(target_bvecs, num_sh):
    target_bvecs = skpp.normalize(target_bvecs, axis=1, norm="l2")

    #output_bvecs = np.concatenate(\
    #  (output_bvecs,target_bvecs,-target_bvecs), axis=1 \
    #)

    wrapper = shmpy.Py_SH_Wrapper()
    output = wrapper.projection_matrix_even(
      num_sh,
      target_bvecs.flatten(),
      target_bvecs.shape
    )

    return np.array(output[0]).reshape(output[1])


#
#
# assumes volume is CHWD
#
def mean_b0s( bvals, bvecs, vol, bval_thold ):
    is_target = bvals < bval_thold
    is_other = bvals >= bval_thold
    output_vol = np.concatenate( 
        (np.mean(vol[is_target,...],axis=0,keepdims=True),
        vol[is_other,...]), axis=0
    )
    bvals = np.concatenate(
        ([0], bvals[is_other]), axis=0
    )
    bvecs = np.concatenate(
        (np.mean(bvecs[is_target,...],axis=0,keepdims=True), bvecs[is_other,...]), axis=0
    )
    return output_vol, bvals, bvecs

#
#
# This function assumes that the bvecs are NOT duplicated on the whole sphere,
# and will copy them to their equal but direction opposite direction.
#
#
# assumes volume is CHWD or CDWH (basically C...)
#
def project_volume( bvals, bvecs, target_bvals, bval_thold, num_sh, volume, wrapper = shmpy.Py_SH_Wrapper()):

    original_shape = volume.shape
    volume = np.reshape(volume, [original_shape[0],-1] )

    num_coefs = wrapper.check_num_even_coef(num_sh)

    N_B0s = np.sum(bvals < bval_thold)

    output_harmonic_volume = np.zeros( \
        [num_coefs * (len(target_bvals)-1) + N_B0s, volume.shape[-1]]\
    ) 

    n_prev = 0
    for idx, t_bval in enumerate(target_bvals):
        is_target = np.abs(bvals - t_bval) < bval_thold

        num_directions = np.sum(is_target)

        if t_bval < bval_thold:
            fit_even_sh = volume[is_target,:]
        else:
            bvecs = skpp.normalize(bvecs[is_target,:], axis=1, norm="l2")
            bvecs_doubled = np.concatenate([bvecs, -bvecs],axis=0)
            bvecs_doubled = np.repeat([bvecs_doubled], volume.shape[-1], axis=0)

            vol_shell = volume[is_target,:]
            vol_shell = np.repeat([vol_shell], volume.shape[-1])

            #TODO: generalize to do whole scan at once
            fit_even_sh = wrapper.fit_even_sh( \
                num_sh,\
                bvecs_doubled.flatten(),\
                vol_shell.flatten(),\
                bvecs_doubled.shape,\
                vol_shell.shape, \
                False
            )
            fit_even_sh = np.reshape(fit_even_sh, [num_coefs, -1])
        output_harmonic_volume[n_prev:num_directions,:] = fit_even_sh #TODO append here
        n_prev += num_directions

    return output_harmonic_volume

#
# assumes NCHWD or NCDWH (basically NC...)
#
# TODO: reorg both output bvals and bvecs
#
#
def reorg_volume(bvals, bvecs, target_bvals, bval_thold, num_sh, vol, pad=True, bval_count_max=None):
    reduced_stack = []
    reduced_bvals = []
    reduced_bvecs = []
    reorder = []
    for idx, t_bval in enumerate(target_bvals):
        is_target = np.abs(bvals - t_bval) < bval_thold

        #make a dictionary
        #unique, counts = np.unique( is_target, return_counts=True )
        #del unique, counts
        num_directions = np.sum(is_target) #dict(zip(unique, counts))[True]

        if num_directions == 0:
            print("bval %0.2f not found with thold %0.2f" % (t_bval, bval_thold))
            exit(1)

        reorder = reorder + np.where(is_target)[0].tolist()
        reduced_stack.append( vol[:,is_target,...] )

        #new bvals
        reduced_bvals.append(bvals[is_target])
        reduced_bvecs.append(bvecs[is_target,:])

        if t_bval == 0:
            continue

        if pad and num_directions < bval_count_max[idx]:
            #pad output img stack
            reduced_stack.append(
                np.zeros(
                    [vol.shape[0]] +
                    [bval_count_max[idx] - num_directions] +
                    list(vol.shape[2:])
                )
            )

            reduced_bvals.append([t_bval for _ in range(bval_count_max[idx] - num_directions)])
            reduced_bvecs.append(np.ones([bval_count_max[idx] - num_directions,3]))

    reduced_stack = np.concatenate( reduced_stack, axis=1 )
    reduced_bvals = np.concatenate( reduced_bvals, axis=0 )
    reduced_bvecs = np.concatenate( reduced_bvecs, axis=0 )
    return reduced_stack, reduced_bvals, reduced_bvecs

##
## Output matrices have dim [NUM DIR] x [SH-COEF]
##
#
# takes in bvals, bvecs, loaded from raw, outputs a re-ordering and a matrix to fit that reordering.
# YOU SHOULD CHECK THE VOLUME to make sure that the volume has the correct number of channels.
# For whatever reason this happens to some dMRI data.
#
# However, this function will pad the output matrix to output zeros for less than a certain number of b-vals
#
# Example:
#   Your data should have [5 b0s] [30 b1000] [30 b2000]
#   Instead, one scan has [5 b0s] [29 b1000] [30 b2000]
#
#   The matrices will always be sized [65=5+30+30] x [NUM_DIR].
#
#   Thus, the outputs will NOT match the inputs, unless you similarly pad the volume.
#
# this function also returns a weighting vector (that does NOT sum to one). It does not need to
# be used, but reweights the outputs so that each approx. b-shell is weighted equally (including b0)
#
# this function does NOT know about the temporal ordering of the outputs
# TODO:...make a function to also output temporal ordering of the outputs for input into the AE
#
def b_transform( bvals, bvecs, target_bvals, bval_thold, num_sh, debug=False, pad=True, bval_count_max=None):

    #reduced_stack = []
    reduced_weight = []
    sh_matrices = []
    orig_sh_matrices = []

    overall_mask = np.array([False for i in range(bvals.shape[0])])
    reorder = []

    for idx, t_bval in enumerate(target_bvals):
        #select the correct bval (up to threshold)
        is_target = np.abs(bvals - t_bval) < bval_thold
        overall_mask = np.logical_or(overall_mask, is_target)

        #make a dictionary
        #unique, counts = np.unique( is_target, return_counts=True )
        #del unique, counts
        num_directions = np.sum(is_target) #dict(zip(unique, counts))[True]

        if num_directions == 0:
            print("bval %0.2f not found with thold %0.2f" % (t_bval, bval_thold))
            exit(1)

        reorder = reorder + np.where(is_target)[0].tolist()

        if num_directions > bval_count_max[idx]:
            print("Error, more than bval_count_max")
            exit(1)

        reduced_weight.append([1.0 / float(num_directions) for i in range(num_directions)])

        #TODO generalize to volumes of bvecs
        if len(bvecs.shape) > 3:
            print("volumes of bvecs not supported")
            exit(1)

        target_bvecs = bvecs[is_target,:]

        if t_bval == 0:
            sh_mat = np.eye(num_directions)
            orig_sh_matrices.append( np.eye(num_directions) )
            put_first = True
        else:
            sh_mat = sh_mat_helper( target_bvecs, num_sh )
            orig_sh_matrices.append( sh_mat )
            put_first = False

        #handles cases with less directions than desired, padding sh_mat with zeros
        if pad and num_directions < bval_count_max[idx]:
            #pad output matrix
            sh_mat = np.concatenate( [sh_mat, np.zeros((bval_count_max[idx] - num_directions, sh_mat.shape[1]))], axis=0)

            #pad weight vector
            reduced_weight.append([0 for i in range(bval_count_max[idx] - num_directions)])

        sh_matrices.append(sh_mat)

    if not np.any(overall_mask):
        print("Warning: Not all b-vals used. Orig vol will not be complete")
        reorder = reorder + np.where(np.logical_not(overall_mask))[0].tolist()

    sh_mat = np.array(block_diag(*sh_matrices))
    reduced_weight_array = np.concatenate(reduced_weight,axis=0)

    orig_sh_mat = np.array(block_diag(*orig_sh_matrices))

    if orig_sh_mat.shape[0] < bvals.shape[0]:
        filler = np.zeros((bvals.shape[0] - orig_sh_mat.shape[0], orig_sh_mat.shape[1]))
        orig_sh_mat = np.concatenate((orig_sh_mat,filler), axis=0)

    order = np.argsort(reorder)
    orig_sh_mat = orig_sh_mat[order,:]
    #reduced_stack = np.concatenate( reduced_stack, axis=3)

    if debug:
        print(reduced_weight)
        print(sh_mat)
        print(sh_mat.shape)
        print(orig_sh_mat.shape)
        print("Output matrices should have dim [NUM DIR] x [SH-COEF]")

    ##
    ## Output matrices have dim [NUM DIR] x [SH-COEF]
    #return sh_mat, reduced_weight_array, reduced_stack, orig_sh_mat
    return sh_mat, reduced_weight_array, orig_sh_mat



