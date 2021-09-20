
import numpy as np
import torch
import joblib


#
# L1 is the "manhattan" distance patch:
#   0 1 0
#   1 1 1
#   0 1 0
# square is the "manhattan" distance patch:
#   1 1 1
#   1 1 1
#   1 1 1
#TODO: generalize to different dimensions
def patch_template(size=1, dim=3, patch_type="L1"):
    if patch_type in ["L1","Linf","square"]:
        return _sym_patch_template(size,dim,patch_type)
    elif patch_type in ["asym_L1","asym_Linf","asym_square"]:
        return _asym_patch_template(size,dim,patch_type)
    else:
        raise Exception("Patch type %s not supported" % patch_type)


def _asym_patch_template( size, dim, patch_type ):
  # if odd sized, just call sym template maker
  if size % 2 == 1:
    return _sym_patch_template(size,dim,patch_type[5:])

  #otherwise it's always an even size
  actual_size = (size - 1) // 2
  actual_dist = (size - 1) / 2
  template = []
  for i in range(-actual_size,actual_size+2):
    for j in range(-actual_size,actual_size+2):
      for k in range(-actual_size,actual_size+2):
        if patch_type == "asym_L1":
          L1_dist = abs(i) + abs(j) + abs(k)
          if L1_dist <= actual_size:
            template.append([i,j,k])
          elif (i > 0 or j > 0 or k > 0) and L1_dist <= actual_size + 1:
            template.append([i,j,k])
        elif patch_type in ["asym_Linf", "asym_square"]:
          template.append([i,j,k])

  template = np.array(template)
  return template


def _sym_patch_template( size, dim, patch_type ):
  template = []
  for i in range(-size,size+1):
    for j in range(-size,size+1):
      for k in range(-size,size+1):
        if patch_type == "L1" and abs(i) + abs(j) + abs(k) <= size:
          template.append([i,j,k])
        elif patch_type in ["Linf", "square"]:
          template.append([i,j,k])

  template = np.array(template)
  return template

class example_in_memory_dataset(torch.utils.data.Dataset):
    def __init__(self, zip_path, idx_pairs, patch_template_instance, rng, scan_type_map, n_per_img=None):

        self.rng = rng
        self.patch_template_instance = patch_template_instance
        print(patch_template_instance.shape)
        #TODO: create center voxel extractor
        #center_vox_extractor = 

        idx_pairs = list(idx_pairs)

        output_harmonic_patches = None
        output_subj_vox = None

        self.sh_mat = {}
        self.subj_space_weights = {}
        self.orig_mat = {}
        self.idx_plus_scan_type = []
        c_vec = []

        for pair_idx, (idx, scan_type) in enumerate(idx_pairs):

            zip_dict = joblib.load(f"{zip_path}/{idx}_{scan_type}.gz")

            #print(A["mask"].shape)
            mask = zip_dict["mask"]
            scan = zip_dict["harmonic_volume"]
            scan_subj_pad = zip_dict["padded_vol"]

            if f"{idx}_{scan_type}" not in self.sh_mat.keys():
                self.sh_mat[f"{idx}_{scan_type}"] = torch.tensor(zip_dict["sh_mat"]).float()
                self.subj_space_weights[f"{idx}_{scan_type}"] = zip_dict["reduced_weight_array"]
                self.orig_mat[f"{idx}_{scan_type}"] = zip_dict["orig_sh_mat"]

            if n_per_img is not None and output_harmonic_patches is None:
                output_harmonic_patches = np.zeros(
                    (n_per_img * len(idx_pairs), scan.shape[1] * patch_template_instance.shape[0])
                )
                output_subj_vox = np.zeros(
                    (n_per_img * len(idx_pairs), scan_subj_pad.shape[1])
                )
                self.n_chan_out_harmonic = scan.shape[1]

            where_output = np.where(mask[0,0])

            perm = self.rng.permutation( np.arange( where_output[0].shape[0] ) )
            where_output = (where_output[0][perm], where_output[1][perm], where_output[2][perm])

            print(scan.shape)

            #print(where_output[0].shape)
            #exit(0)
            for patch_idx, index in enumerate(list(zip(*where_output))):

                #TODO: check exit condition where num where_output indices is LESS than n_per_img
                #TODO: and truncate accordingly
                if n_per_img is not None and patch_idx >= n_per_img:
                    break

                #if patch_idx % 2000 == 0:
                #    print(patch_idx)

                patch_indices = index + patch_template_instance

                values = scan[
                    0,:,
                    patch_indices[:,0], \
                    patch_indices[:,1], \
                    patch_indices[:,2]
                ].flatten()

                #values = values[np.newaxis,:]
                if n_per_img is not None:
                    output_harmonic_patches[patch_idx + n_per_img * pair_idx,:] = values
                    output_subj_vox[patch_idx + n_per_img * pair_idx,:] = scan_subj_pad[0,:,index[0],index[1],index[2]]
                elif output_harmonic_patches is not None:
                    output_harmonic_patches = np.concatenate((output_harmonic_patches,values),axis=0)
                    output_subj_vox = np.concatenate((output_subj_vox,scan_subj_pad[0,:,index[0],index[1],index[2]]),axis=0)
                else:
                    output_harmonic_patches = values
                    output_subj_vox = scan_subj_pad[0,:,index[0],index[1],index[2]]

                self.idx_plus_scan_type.append(f"{idx}_{scan_type}")
                c_vec.append(scan_type_map[f"{scan_type}"])

            print(output_harmonic_patches.shape)
            print(output_subj_vox.shape)

        self.c_vec = torch.tensor(np.array(c_vec)[:,np.newaxis]).float()
        self.output_harmonic_patches = torch.tensor(output_harmonic_patches).float()
        self.output_subj_vox = torch.tensor(output_subj_vox).float()
        return

    def __getitem__(self,idx):
        return \
            self.output_harmonic_patches[idx], \
            self.output_subj_vox[idx], \
            self.sh_mat[self.idx_plus_scan_type[idx]], \
            self.subj_space_weights[self.idx_plus_scan_type[idx]], \
            self.c_vec[idx]

    def get_center_voxel_function(self):
        n_before = self.n_chan_out_harmonic * ( self.patch_template_instance.shape[0] // 2 )
        return lambda vol : vol[:,n_before:(n_before + self.n_chan_out_harmonic)]

    def __len__(self):
        return len(self.idx_plus_scan_type)

if __name__ == "__main__":
    example_in_memory_dataset(
        zip_path="data/zips/",
        idx_pairs=zip(2*list(range(10)),10*["1200"] + 10*["7T"]),
        patch_template_instance=patch_template(1,3),
        rng = np.random.default_rng(1919),
        n_per_img=100000
    )







