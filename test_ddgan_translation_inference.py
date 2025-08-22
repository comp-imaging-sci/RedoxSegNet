# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np

import os

import torchvision
from score_sde.models.ncsnpp_generator_adagn import NCSNpp


from datasets_prep.microscopy_datasets import microscopy_new_inference
from tifffile import imwrite
#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one


#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
def sample_posterior(coefficients, x_0,x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos




def sample_from_model(coefficients, generator, n_time, x_init, T, opt,source):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x


def sample_from_model_ddim(coefficients, generator, n_time, x_init, T, opt,eta,source):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
          
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior_ddim(coefficients, x_0[:,[0],:], x, t,eta)
            x = x_new.detach()
        
    return x


def sample_posterior_ddim(coefficients, x_0,x_t, t,eta):
    def predict_eps_from_xstart(x_0, x_t, t):
        eps = (
            extract(coefficients.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t-x_0
        )/extract(coefficients.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return eps 
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def ddim_sample(x_0, x_t, t,eta):
        eps = predict_eps_from_xstart(x_0, x_t, t)

        alpha_bar = extract(coefficients.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = extract(coefficients.alphas_cumprod_prev, t, x_t.shape)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        mean = x_0 * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps


        # _, _, log_var = q_posterior(x_0, x_t, t)
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] *sigma* noise
            
    sample_x_pos = ddim_sample(x_0, x_t, t,eta)
    
    return sample_x_pos





#%%
def sample_and_test(args):
    device = torch.device('cuda:0')

    torch.manual_seed(42)


    dataset = microscopy_new_inference(dataroot=args.data_dir,phase='test',input_selection=args.input_selection,input_channels=args.input_channels,sum_input=args.sum_input)



    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True)


    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    
    netG = NCSNpp(args).to(device)
    ckpt = torch.load(args.checkpoint_model, map_location=device)
    
    #loading weights from ddp in single gpu
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    
    T = get_time_schedule(args, device)
    coeff = Diffusion_Coefficients(args, device)
    # print(coeff)
    pos_coeff = Posterior_Coefficients(args, device)
    # print(pos_coeff)
        

    im_res = np.zeros([len(dataset),args.num_target_channels,args.image_size, args.image_size])
    
    # save_dir = "./generated_samples/{}".format(args.dataset)
    save_dir = args.result_dir
    im_save_path_tif = os.path.join(save_dir, 'images_test_tif')
    os.makedirs(im_save_path_tif,exist_ok = True)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    total_im=0
    for iteration, (x_val , y_val,names_val) in enumerate(data_loader):
        # real_data = x_val.to(device, non_blocking=True)
        source_data = x_val.to(device, non_blocking=True)

        x_t_1 = torch.randn(source_data.shape[0], args.num_target_channels,args.image_size, args.image_size).to(device)
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args,source_data)
        

        fake_sample = to_range_0_1(fake_sample) #; fake_sample = fake_sample/fake_sample.mean()
        
        source_data_save = to_range_0_1(source_data) #; real_data = real_data/real_data.mean()
        B,C,H,W=source_data_save.shape
        source_data_save=(source_data_save.permute(0,2,1,3).reshape([B,1,H,W*C]))

        save_im = torch.cat((source_data_save, fake_sample),axis=-1)
        torchvision.utils.save_image(save_im, os.path.join(save_dir, 'sample_discrete_batch_{}.png'.format(iteration)), normalize=True,nrow=1)

        fake_sample=fake_sample.cpu().numpy().squeeze()
       

        batch_size_cur= x_val.shape[0]
       

        im_res[total_im:total_im+batch_size_cur,] = fake_sample
  
        total_im=total_im+batch_size_cur

        for ind_temp in range(batch_size_cur):
            save_dir_tif = os.path.join(im_save_path_tif, names_val[ind_temp])
            imwrite(save_dir_tif,fake_sample[ind_temp])


   
    np.save(save_dir+'/results.npy',im_res) 

    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                            help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int,default=1000)
    parser.add_argument('--num_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                            help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1,
                            help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                            help='beta_max for diffusion')
    
    
    parser.add_argument('--num_channels_dae', type=int, default=128,
                            help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                            help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                            help='channel multiplier')

    parser.add_argument('--num_res_blocks', type=int, default=2,
                            help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                            help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                            help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                            help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                            help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                            help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                            help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                            help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                            help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                            help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                            help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    
    #geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    

    parser.add_argument('--dataset', default='Numpy_data_microscopy', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=256,
                            help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=200, help='sample generating batch size')
    parser.add_argument('--checkpoint_dir', type=str, default='./',
                        help='path for checkpoints and model savings')

    parser.add_argument('--degradation_type', type=str, default='GaussianNoise', help='type of degradation process')
    parser.add_argument('--mask_dir', type=str, default='./', help='path_contains mask')
    parser.add_argument('--input_selection',action='store_true', default=False,help='select specific channels in source data')
    parser.add_argument('--input_channels', nargs='+', type=int, default=[0], help='channel selection from source data')
    parser.add_argument('--sum_input',action='store_true', default=False,help='sum all the source channel and feed them as a single channel input')
    parser.add_argument('--num__source_channels', type=int, default=3,
                            help='channel of image')
    parser.add_argument('--num_target_channels', type=int, default=1,
                            help='channel of image')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='path_contains data')
    parser.add_argument('--result_dir', type=str, default='./',
                        help='path for results')
    parser.add_argument('--checkpoint_model', type=str, default='checkpoints/mitechondria_translation_new_data_1520_ddgan_timestep4_input_channel_1/netG_100.pth',
                        help='path for model weight')
        



   
    args = parser.parse_args()
    
    sample_and_test(args)
    
   
                