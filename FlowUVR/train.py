import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import gc
import copy

import torch
print(torch.__version__)
torch.backends.cuda.matmul.allow_tf32 = False

from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from FlowUVR import CycleOTFlow, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils import *
from my_utils.dino_struct import DinoStructureLoss
from dataset import UnpairedDataset
from temploss import *

def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint)
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"resume_from_checkpoint: {resume_from_checkpoint}")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision,
                                              use_fast=False,local_files_only=True)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet,
                                                                                                  return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    crit_cycle, crit_fm, crit_ke ,crit_div = PSNRLoss(), torch.nn.L1Loss(), torch.nn.MSELoss(), nn.MSELoss()

    ms_ssim_criterion = MS_SSIM_Loss(data_range=2.0).to(accelerator.device)

    temploss_cyc = ReconstructionTemporalLoss(
        lambda_rec=0.5,
        num_frames=args.num_frames)
    temploss_cyc = temploss_cyc.to(accelerator.device)
    temploss_gen = GenerationTemporalLoss(
        lambda_gen=0.5,
        num_frames=args.num_frames,
        raft_pretrained=True
    )
    temploss_gen = temploss_gen.to(accelerator.device)


    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleOTFlow.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        print(f"resume_from_checkpoint: {resume_from_checkpoint}")

        if os.path.isfile(resume_from_checkpoint):
            print(f"resume_from_checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')

            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(unet, checkpoint['sd_encoder'], adapter_name="default_encoder")
            set_peft_model_state_dict(unet, checkpoint['sd_decoder'], adapter_name="default_decoder")
            set_peft_model_state_dict(unet, checkpoint['sd_other'], adapter_name="default_others")

            vae_enc.load_state_dict(checkpoint['sd_vae_enc'])
            vae_dec.load_state_dict(checkpoint['sd_vae_dec'])

            if 'global_step' in checkpoint:
                global_step = checkpoint['global_step']
                print(f"global_step: {global_step}")
            if 'epoch' in checkpoint:
                first_epoch = checkpoint['epoch'] + 1
                print(f"epoch: {first_epoch}")

            print(f"Successful")
        else:
            print(f"warning: {resume_from_checkpoint}")

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    dataset_train = UnpairedDataset(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train",
                                    tokenizer=tokenizer, num_frames=args.num_frames, samples_per_video=100)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.dataloader_num_workers)

    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt

    test_A_path = os.path.join(args.dataset_folder, "test_A")
    test_B_path = os.path.join(args.dataset_folder, "test_B")

    video_folders_src_test = [f for f in os.listdir(test_A_path) if os.path.isdir(os.path.join(test_A_path, f))]
    video_folders_tgt_test = [f for f in os.listdir(test_B_path) if os.path.isdir(os.path.join(test_B_path, f))]


    l_images_src_test = get_all_test_images(test_A_path)
    l_images_tgt_test = get_all_test_images(test_B_path)


    # make the reference FID statistics
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        """
        FID reference statistics for A -> B translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)


        for video_folder in tqdm(video_folders_tgt_test):

            ref_video_folder = os.path.join(output_dir_ref, video_folder)
            os.makedirs(ref_video_folder, exist_ok=True)

            video_path = os.path.join(test_B_path, video_folder)
            frames = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                frames.extend(glob.glob(os.path.join(video_path, ext)))
            frames = sorted(frames)

            for frame_path in frames:
                frame_name = os.path.basename(frame_path)
                if frame_name.lower().endswith('.jpg') or frame_name.lower().endswith('.jpeg'):
                    outf = os.path.join(ref_video_folder, frame_name.replace(".jpg", ".png").replace(".jpeg", ".png"))
                else:
                    outf = os.path.join(ref_video_folder, frame_name)

                if not os.path.exists(outf):
                    try:
                        _img = T_val(Image.open(frame_path).convert("RGB"))
                        _img.save(outf)
                    except Exception as e:
                        print(f"处理图像 {frame_path} 时出错: {e}")
                        continue

        try:
            ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                                               shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                               mode="clean", custom_fn_resize=None, description="", verbose=True,
                                               custom_image_tranform=None)
            a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
            print(f"True:a2b FID，use {len(ref_features)} images")
        except Exception as e:
            print(f"error: {e}")
            a2b_ref_mu, a2b_ref_sigma = None, None

        """
        FID reference statistics for B -> A translation
        """
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)

        for video_folder in tqdm(video_folders_src_test):
            ref_video_folder = os.path.join(output_dir_ref, video_folder)
            os.makedirs(ref_video_folder, exist_ok=True)
            video_path = os.path.join(test_A_path, video_folder)
            frames = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                frames.extend(glob.glob(os.path.join(video_path, ext)))
            frames = sorted(frames)

            for frame_path in frames:
                frame_name = os.path.basename(frame_path)
                if frame_name.lower().endswith('.jpg') or frame_name.lower().endswith('.jpeg'):
                    outf = os.path.join(ref_video_folder, frame_name.replace(".jpg", ".png").replace(".jpeg", ".png"))
                else:
                    outf = os.path.join(ref_video_folder, frame_name)

                if not os.path.exists(outf):
                    try:
                        _img = T_val(Image.open(frame_path).convert("RGB"))
                        _img.save(outf)
                    except Exception as e:
                        print(f"process image {frame_path} error: {e}")
                        continue

        # 计算参考图像的特征
        try:
            # 使用 get_folder_features 并指定递归搜索
            ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                                               shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                               mode="clean", custom_fn_resize=None, description="", verbose=True,
                                               custom_image_tranform=None)
            b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
            print(f"True:b2a FID，use {len(ref_features)} images")
        except Exception as e:
            print(f"error: {e}")
            b2a_ref_mu, b2a_ref_sigma = None, None

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps * accelerator.num_processes,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)


    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)

    optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )


    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    print(f"training params:")
    print(f"  - max_train_steps: {args.max_train_steps}")
    print(f"  - max_train_epochs: {args.max_train_epochs}")
    print(f"  - gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  - train_batch_size: {args.train_batch_size}")
    print(f"  - dataset: {len(dataset_train)}")
    print(f"  - dataloader (each epoch steps): {len(train_dataloader)}")
    actual_total_steps = len(train_dataloader) * args.max_train_epochs // args.gradient_accumulation_steps
    print(f"  - actual_total_steps: {actual_total_steps}")

    if actual_total_steps < args.max_train_steps:
        required_epochs = (args.max_train_steps * args.gradient_accumulation_steps) // len(train_dataloader) + 1
        args.max_train_epochs = max(args.max_train_epochs, required_epochs)
        print(f"max_train_epochs: {args.max_train_epochs}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    def v_func(x):
        return CycleOTFlow.forward_with_networks(x,  unet, noise_scheduler_1step, timesteps, fixed_a2b_emb)

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                scaling_factor = 0.18215
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                B, T = img_a.shape[0], img_a.shape[1]
                img_a = img_a.reshape(B * T, *img_a.shape[2:])
                img_b = img_b.reshape(B * T, *img_b.shape[2:])
                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz,
                                         device=img_a.device).long()
                img_a_enc = vae_enc(img_a, direction="a2b").to(img_a.dtype) * scaling_factor
                img_b_enc = vae_enc(img_b, direction="b2a").to(img_b.dtype) * scaling_factor
                t = torch.rand(B * T, device=img_a_enc.device)
                t_broadcast = t[:, None, None, None]
                img_a_enc = (1 - t_broadcast) * img_a_enc + t_broadcast * img_b_enc
                img_b_enc = (1 - t_broadcast) * img_b_enc + t_broadcast * img_a_enc

                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                cyc_v_fake_a2b = CycleOTFlow.forward_with_networks(img_a_enc, unet, noise_scheduler_1step, timesteps,
                                                                   fixed_a2b_emb)
                cyc_fake_b = vae_dec((cyc_v_fake_a2b + img_a_enc) / scaling_factor, direction="a2b")
                cyc_fake_b_enc = vae_enc(cyc_fake_b, direction="b2a").to(img_a.dtype) * scaling_factor
                cyc_v_rec_b2a = CycleOTFlow.forward_with_networks(cyc_fake_b_enc, unet, noise_scheduler_1step,
                                                                  timesteps, fixed_b2a_emb)
                cyc_rec_a = vae_dec((cyc_v_rec_b2a + cyc_fake_b_enc) / scaling_factor, direction="b2a")
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += ms_ssim_criterion(cyc_rec_a, img_a) * args.lambda_cycle_ms_ssim
                loss_temp_rec_a = temploss_cyc(cyc_rec_a, img_a) * args.lambda_temp_cyc

                # B -> fake A -> rec B
                cyc_v_fake_b2a = CycleOTFlow.forward_with_networks(img_b_enc, unet, noise_scheduler_1step, timesteps,
                                                                   fixed_b2a_emb)
                cyc_fake_a = vae_dec((cyc_v_fake_b2a + img_b_enc) / scaling_factor, direction="b2a")
                cyc_fake_a_enc = vae_enc(cyc_fake_a, direction="a2b").to(img_a.dtype) * scaling_factor

                cyc_v_rec_a2b = CycleOTFlow.forward_with_networks(cyc_fake_a_enc, unet, noise_scheduler_1step,
                                                                  timesteps, fixed_a2b_emb)
                cyc_rec_b = vae_dec((cyc_v_rec_a2b + cyc_fake_a_enc) / scaling_factor, direction="a2b")

                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += ms_ssim_criterion(cyc_rec_b, img_b) * args.lambda_cycle_ms_ssim
                loss_temp_rec_b = temploss_cyc(cyc_rec_b, img_b) * args.lambda_temp_cyc

                accelerator.backward(loss_cycle_a + loss_cycle_b + loss_temp_rec_a + loss_temp_rec_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                img_a_enc_gan = vae_enc(img_a, direction="a2b").to(img_a.dtype) * scaling_factor
                img_b_enc_gan = vae_enc(img_b, direction="b2a").to(img_b.dtype) * scaling_factor

                img_a_enc_gan = (1 - t_broadcast) * img_a_enc_gan + t_broadcast * img_b_enc_gan
                img_b_enc_gan = (1 - t_broadcast) * img_b_enc_gan + t_broadcast * img_a_enc_gan

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                v_fake_b2a = CycleOTFlow.forward_with_networks(img_b_enc_gan, unet, noise_scheduler_1step, timesteps,
                                                               fixed_b2a_emb)
                fake_a = vae_dec((v_fake_b2a + img_b_enc_gan) / scaling_factor, direction="b2a")
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                v_fake_a2b = CycleOTFlow.forward_with_networks(img_a_enc_gan, unet, noise_scheduler_1step, timesteps,
                                                               fixed_a2b_emb)
                fake_b = vae_dec((v_fake_a2b + img_a_enc_gan) / scaling_factor, direction="a2b")
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_temp_gen_b = temploss_gen(fake_b) * args.lambda_temp_gen
                loss_temp_gen_a = temploss_gen(fake_a) * args.lambda_temp_gen
                accelerator.backward(loss_gan_a + loss_gan_b + loss_temp_gen_a + loss_temp_gen_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                img_a_enc_fm = vae_enc(img_a, direction="a2b").to(img_a.dtype) * scaling_factor
                img_b_enc_fm = vae_enc(img_b, direction="b2a").to(img_b.dtype) * scaling_factor
                img_a_enc_fm = (1 - t_broadcast) * img_a_enc_fm + t_broadcast * img_b_enc_fm
                img_b_enc_fm = (1 - t_broadcast) * img_b_enc_fm + t_broadcast * img_a_enc_fm

                """
                OTFlow  
                """

                v_a2b = CycleOTFlow.forward_with_networks(img_a_enc_fm, unet, noise_scheduler_1step, timesteps,
                                                          fixed_a2b_emb)
                # divergence penalty (Hutchinson)
                div_a2b = divergence_hutchinson_fd(v_func, img_a_enc_fm)
                loss_div_a2b = crit_div(div_a2b, torch.zeros_like(div_a2b))
                v_b2a = CycleOTFlow.forward_with_networks(img_b_enc_fm, unet, noise_scheduler_1step,
                                                          timesteps, fixed_b2a_emb)
                div_b2a = divergence_hutchinson_fd(v_func, img_b_enc_fm)
                loss_div_b2a = crit_div(div_b2a, torch.zeros_like(div_b2a))
                loss_div = loss_div_a2b + loss_div_b2a
                # kinetic energy
                v_norm_a2b = torch.norm(v_a2b, dim=1)
                loss_ke_a2b = crit_ke(v_norm_a2b, torch.zeros_like(v_norm_a2b))
                v_norm_b2a = torch.norm(v_b2a, dim=1)
                loss_ke_b2a = crit_ke(v_norm_b2a, torch.zeros_like(v_norm_b2a))
                loss_ke = loss_ke_a2b + loss_ke_b2a
                loss_V_zero = crit_fm(v_a2b + v_b2a, torch.zeros_like(v_a2b))
                loss_FM = loss_V_zero * args.lambda_fm + loss_ke * args.lambda_ke + loss_div
                accelerator.backward(loss_FM, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["temp_cyc_a"] = loss_temp_rec_a.detach().item()
            logs["temp_cyc_b"] = loss_temp_rec_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["loss_ke"] = loss_ke.detach().item()
            logs["loss_div"] = loss_div.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["fm_zero"] = loss_V_zero.detach().item()


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    steps = 1
                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 1:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1,
                                                  device="cuda").long()
                        net_dino = DinoStructureLoss()

                        """
                        Evaluate "A->B"
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)

                        for video_folder in video_folders_src_test:
                            output_video_folder = os.path.join(fid_output_dir, video_folder)
                            os.makedirs(output_video_folder, exist_ok=True)

                        l_dino_scores_a2b = []
                        # get val input images from domain a
                        processed_count = 0
                        for video_folder in video_folders_src_test:
                            if args.validation_num_images > 0 and processed_count >= args.validation_num_images:
                                break

                            video_path = os.path.join(test_A_path, video_folder)
                            frames = []
                            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                                frames.extend(glob.glob(os.path.join(video_path, ext)))
                            frames = sorted(frames)

                            for idx, input_img_path in enumerate(frames):
                                if args.validation_num_images > 0 and processed_count >= args.validation_num_images:
                                    break

                                outf = os.path.join(fid_output_dir, video_folder, f"{idx:04d}.png")
                                with torch.no_grad():
                                    input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                    img_a = transforms.ToTensor()(input_img)
                                    img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()

                                    img_a_padded, original_shape = pad_to_multiple(img_a, multiple=16)
                                    scaling_factor = 0.18215
                                    img_a_enc = vae_enc(img_a_padded, direction="a2b").to(img_a.dtype) * scaling_factor
                                    for step in range(steps):
                                        v_eval_fake_a2b = CycleOTFlow.forward_with_networks(img_a_enc, eval_unet,
                                                                                             noise_scheduler_1step,
                                                                                             _timesteps,
                                                                                             fixed_a2b_emb[0:1])
                                        img_a_enc = img_a_enc + v_eval_fake_a2b / steps
                                    img_a_enc = img_a_enc / scaling_factor
                                    eval_fake_b_padded = eval_vae_dec(img_a_enc, direction="a2b")
                                    eval_fake_b = crop_to_original(eval_fake_b_padded, original_shape)
                                    # Clamp values to valid range [0, 1] to prevent color wrapping
                                    eval_fake_b_tensor = (eval_fake_b[0] * 0.5 + 0.5).clamp(0, 1).cpu()

                                    eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b_tensor)
                                    eval_fake_b_pil.save(outf)

                                    a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                    b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                    dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                    l_dino_scores_a2b.append(dino_ssim)
                                    processed_count += 1

                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                                                           shuffle=False, seed=0, batch_size=8,
                                                           device=torch.device("cuda"),
                                                           mode="clean", custom_fn_resize=None, description="",
                                                           verbose=True,
                                                           custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")

                        """
                        compute FID for "B->A"
                        """
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_b2a")
                        os.makedirs(fid_output_dir, exist_ok=True)

                        for video_folder in video_folders_tgt_test:
                            output_video_folder = os.path.join(fid_output_dir, video_folder)
                            os.makedirs(output_video_folder, exist_ok=True)

                        l_dino_scores_b2a = []
                        # get val input images from domain b
                        processed_count = 0
                        for video_folder in video_folders_tgt_test:
                            if args.validation_num_images > 0 and processed_count >= args.validation_num_images:
                                break

                            video_path = os.path.join(test_B_path, video_folder)
                            frames = []
                            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                                frames.extend(glob.glob(os.path.join(video_path, ext)))
                            frames = sorted(frames)

                            for idx, input_img_path in enumerate(frames):
                                if args.validation_num_images > 0 and processed_count >= args.validation_num_images:
                                    break

                                outf = os.path.join(fid_output_dir, video_folder, f"{idx:04d}.png")
                                with torch.no_grad():
                                    input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                    img_b = transforms.ToTensor()(input_img)
                                    img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()

                                    img_b_padded, original_shape = pad_to_multiple(img_b, multiple=16)
                                    scaling_factor = 0.18215
                                    img_b_enc = vae_enc(img_b_padded, direction="b2a").to(img_b.dtype) * scaling_factor
                                    for step in range(steps):
                                        v_eval_fake_b2a = CycleOTFlow.forward_with_networks(img_b_enc, eval_unet,
                                                                                             noise_scheduler_1step,
                                                                                             _timesteps,
                                                                                             fixed_b2a_emb[0:1])
                                        img_b_enc = img_b_enc + v_eval_fake_b2a / steps
                                    img_b_enc = img_b_enc / scaling_factor
                                    eval_fake_a_padded = eval_vae_dec(img_b_enc, direction="b2a")


                                    eval_fake_a = crop_to_original(eval_fake_a_padded, original_shape)

                                    eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                    eval_fake_a_pil.save(outf)

                                    a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                    b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                    dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                    l_dino_scores_b2a.append(dino_ssim)
                                    processed_count += 1

                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                                                           shuffle=False, seed=0, batch_size=8,
                                                           device=torch.device("cuda"),
                                                           mode="clean", custom_fn_resize=None, description="",
                                                           verbose=True,
                                                           custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        print(f"step={global_step}, fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}")
                        logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                        del net_dino  # free up memory

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
