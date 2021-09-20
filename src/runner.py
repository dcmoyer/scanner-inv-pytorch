

import torch
import loader
import arch
import losses
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

import argparse

parser = argparse.ArgumentParser(description=\
    "runs inv-rep auto-encoder training"
)

parser.add_argument("--hcp-zip-path")
parser.add_argument("--save-path",default=None)

args = parser.parse_args()

PATH_TO_HCP_DATA=args.hcp_zip_path
save_path=args.save_path

n_epochs = 10000
n_adv_per_enc = 1 #critic index
burnin=5 #n_epochs for the adversary
LR=1e-4
adv_LR=1e-4
batch_size=128
save_freq=5

scan_type_map = {
    "1200" : 0,
    "7T" : 1
}

#train_iterator = loader.example_in_memory_dataset(
#    zip_path=f"{PATH_TO_HCP_DATA}",
#    idx_pairs=zip(2*list(range(10)),10*["1200"] + 10*["7T"]),
#    patch_template_instance=loader.patch_template(1,3),
#    scan_type_map=scan_type_map, rng = np.random.default_rng(1919),
#    n_per_img=100000,
#)

#train_iterator = loader.example_in_memory_dataset(
#    zip_path=f"{PATH_TO_HCP_DATA}",
#    idx_pairs=zip([0,0],["1200"] + ["7T"]),
#    patch_template_instance=loader.patch_template(1,3),
#    scan_type_map=scan_type_map, rng = np.random.default_rng(1919),
#    n_per_img=10000,
#)

#val_iterator = loader.example_in_memory_dataset(
#    zip_path=f"{PATH_TO_HCP_DATA}",
#    idx_pairs=zip(2*list(range(10,20)),10*["1200"] + 10*["7T"]),
#    patch_template_instance=patch_template(1,3),
#    rng = np.random.default_rng(1919),
#    n_per_img=100000
#)
#
#test_iterator = loader.example_in_memory_dataset(
#    zip_path=f"{PATH_TO_HCP_DATA}",
#    idx_pairs=zip(2*list(range(20,30)),10*["1200"] + 10*["7T"]),
#    patch_template_instance=patch_template(1,3),
#    rng = np.random.default_rng(1919),
#    n_per_img=100000
#)

train_loader = torch.utils.data.DataLoader(
    train_iterator,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

#val_loader = torch.utils.data.DataLoader(
#    val_iterator,
#    batch_size=batch_size,
#    shuffle=True,
#    pin_memory=True
#)
#
#test_loader = torch.utils.data.DataLoader(
#    test_iterator,
#    batch_size=batch_size,
#    shuffle=True,
#    pin_memory=True
#)

center_vox_func = train_iterator.get_center_voxel_function()

enc_obj = arch.encoder( 322, 32 )
dec_obj = arch.decoder( 32, 322, 1 )
adv_obj = arch.adv( 322, 1 )

enc_obj.to(device)
dec_obj.to(device)
adv_obj.to(device)

#should use itertools chain
optimizer = torch.optim.Adam(
    list(enc_obj.parameters()) + list(dec_obj.parameters()), lr=LR
)
adv_optimizer = torch.optim.Adam(adv_obj.parameters(), lr=adv_LR)

loss_weights = {
    "recon" : 1.0,\
    "prior" : 1.0,\
    "projection" : 1.0,\
    "marg" : 0.01,\
    "adv" : 10.0\
}

for epoch in range(n_epochs):

    train_loss = 0
    adv_loss = 0
    n = 0
    n_adv = 0

    total_recon_loss = 0
    total_kl_loss = 0
    total_proj_loss = 0
    total_marg_loss = 0
    total_adv_loss = 0

    for d_idx,batch in enumerate(train_loader):
        #print(f"batch {d_idx}", flush=True)

        x = batch[0]
        x_subj_space = batch[1]
        sh_mat = batch[2]
        sh_weights = batch[3]
        c = batch[4]

        x = x.to(device)
        x_subj_space = x_subj_space.to(device)
        sh_mat = sh_mat.to(device)
        sh_weights = sh_weights.to(device)
        c = c.to(device)

        if epoch < burnin or d_idx % (n_adv_per_enc+1) > 0:
            adv_optimizer.zero_grad()
            
            loss = losses.adv_training_step(
                enc_obj, dec_obj, adv_obj, x, c
            )

            loss.backward(retain_graph=True)
            adv_optimizer.step()
            adv_loss += loss.item()*x.size()[0]
            n_adv += x.size()[0]

        else:

            optimizer.zero_grad()

            loss, (recon_loss,kl_loss, proj_loss, marg_loss, gen_adv_loss) = losses.enc_dec_training_step(
                enc_obj, dec_obj, adv_obj,
                x, c, center_vox_func,  x_subj_space, sh_mat, sh_weights,
                loss_weights, 32
            )

            total_recon_loss += recon_loss.item()*x.size()[0]
            total_kl_loss += kl_loss.item()*x.size()[0]
            total_proj_loss += proj_loss.item()*x.size()[0]
            total_marg_loss += marg_loss.item()*x.size()[0]
            total_adv_loss += gen_adv_loss.item()*x.size()[0]

            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()*x.size()[0]
            n += x.size()[0]

        del x, x_subj_space, sh_mat, sh_weights, c

    if save_path is not None and epoch > burnin and epoch % save_freq == 0:
        torch.save(
            {
                "enc":enc_obj.state_dict(),
                "dec":dec_obj.state_dict(),
                "adv":adv_obj.state_dict()
            },
            f"{save_path}/{epoch}.pth"
        )

    if epoch > burnin:
        print("epoch",epoch)
        print("train loss total",train_loss / n)
        print("adv loss total",adv_loss / n_adv)
        print(
            "train loss, recon", total_recon_loss / n,
            "kl", total_kl_loss/ n,
            "proj", total_proj_loss/ n,
            "marg", total_marg_loss/ n,
            "adv", total_adv_loss/ n
        )






