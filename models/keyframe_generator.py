import copy
import cv2
import numpy as np
import torch
import skimage

from einops import rearrange
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2
from pytorch3d.render import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds
from utils.midas_utils import dpt_transform, dpt_512_transform
from utils.utils import functbl


class FrameSyn(nn.Module):
    def __init__(
        self, 
        config, 
        inpainting_pipeline, 
        depth_model, 
        vae,
        rotation, 
        image, 
        inpainting_prompt,
        adaptive_negative_prompt
    ):
        super().__init__()

        self.device = config['device']
        self.config = config
        self.background_hard_depth = config['depth_shift'] + config['fg_depth_range']
        self.is_upper_mask_aggressive = False
        self.use_noprompt = False
        self.total_frames = config['frames']

        self.inpainting_prompt = inpainting_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt
        self.inpainting_pipeline = inpainting_pipeline

        # resize image to 512x512
        image = image.resize((512, 512))
        self.image_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])(image).unsqueeze(0).to(self.device)

        self.depth_model = depth_model
        with torch.no_grad():
            self.depth, self.disparity = self.get_depth(self.image_tensor)

        self.current_camera = self.get_init_camera()
        if self.config['motion'] == 'rotations':
            self.current_camera.rotating = rotation != 0
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 0
            self.current_camera.rotating_right = rotation
            self.current_camera.move_dir = torch.tensor([[-config['right_multiplier'], 0.0, -config['forward_speed_multiplier']]], device=self.device)
        elif self.config['motion'] == 'predefined':
            intrinsics = np.load(self.config['intrinsics']).astype(np.float32)
            extrinsics = np.load(self.config['extrinsics']).astype(np.float32)

            intrinsics = torch.from_numpy(intrinsics).to(self.device)
            extrinsics = torch.from_numpy(extrinsics).to(self.device)
        
            # Extend intrinsics to 4x4 with zeros and assign 1 to the last row and column as required by the camera class
            Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
            Ks[:, 2, 3] = Ks[:, 3, 2] = 1

            Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]

            # PerspectiveCamera operate on row-vector matrices while the loaded extrinsics are column-vector matrices
            Rs = Rs.movedim(1, 2)

            self.predefined_cameras = [
                PerspectiveCameras(K=K.unsqueeze(0), R=R.T.unsqueeze(0), T=t.unsqueeze(0), device=self.device)
                for K, R, t in zip(Ks, Rs, ts)
            ]
            self.current_camera = self.predefined_cameras[0]

        self.images = [self.image_tensor]
        self.inpaint_input_image = [image]
        self.disparities = [self.disparity]
        self.depths = [self.depth]
        self.masks = [torch.ones_like(self.depth)]
        self.post_masks = [torch.ones_like(self.depth)]
        self.post_mask_tmp = None
        self.rendered_images = [self.image_tensor]
        self.rendered_depths = [self.depth]

        self.vae = vae
        self.decoder_copy = copy.deepcopy(self.vae.decoder)

        self.camera_speed = self.config['camera_speed'] if rotation == 0 else self.config['camera_speed'] * self.config['camera_speed_multiplier_rotation']
        self.camera = [self.current_camera]

        # create mask for inpainting of the right size, white area around the image in the middle
        inpainting_resolution = self.config['inpainting_resolution']
        self.border_mask = torch.ones(
            (1, 1, inpainting_resolution, inpainting_resolution)
        ).to(self.device)
        self.border_size = (inpainting_resolution - 512) // 2
        self.border_mask[:, :, self.border_size: -self.border_size, self.border_size:-self.border_size] = 0
        self.border_image = torch.zeros(
            1, 3, inpainting_resolution, inpainting_resolution
        ).to(self.device)
        self.images_orig_decoder = [
            v2.Resize((inpainting_resolution, inpainting_resolution))(self.image_tensor)
        ]

        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, 'h w c -> (h w) c').to(self.device)

        self.kf_delta_t = self.camera_speed

    def get_depth(self, image):
        if self.depth_model is None:
            depth = torch.zeros_like(image[:, 0:1])
            disparity = torch.zeros_like(image[:, 0:1])
            return depth, disparity
        if self.config['depth_model'].lower() == 'midas':
            # MiDaS
            disparity = self.depth_model(dpt_transform(image))
            disparity = F.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.config['depth_model'].lower() == 'midas_v3.1':
            disparity = self.depth_model(dpt_512_transform(image))
            disparity = F.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        elif self.config['depth_model'].lower() == 'zoedepth':
            depth = self.depth_model(image)['metric_depth']
        depth = depth + self.config['depth_shift']
        disparity = 1 / depth
        return depth, disparity
    
    
    def get_init_camera(self):
        """
        K = [
            [fx,   0,   256,   0],
            [0,    fy,  256,   0],
            [0,    0,       0,   1],
            [0,    0,       1,   0]
        ]
        """
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.config['init_focal_length']
        K[0, 1, 1] = self.config['init_focal_length']
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return camera
    
    def finetune_depth_model_step(self, target_depth, inpainted_image, mask_align=None, mask_cutoff=None, cutoff_depth=None):
        next_depth, _ = self.get_depth(inpainted_image.detach().cuda())
        
        # L1 loss for the mask_align region
        loss_align = F.l1_loss(target_depth.detach(), next_depth, reduction='none')
        if mask_align is not None and torch.any(mask_align):
            mask_align = mask_align.detach()
            loss_align = (loss_align * mask_align)[mask_align > 0].mean()
        else:
            loss_align = torch.zeros(1).to(self.device)

        # Hinge loss for the mask_cutoff region
        if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
            hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
            hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction='none')
            mask_cutoff = mask_cutoff.detach()
            hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
        else:
            hinge_loss = torch.zeros(1).to(self.device)

        total_loss = loss_align + hinge_loss
        if torch.isnan(total_loss):
            raise ValueError("Depth FT loss is NaN")
        # print both losses and total loss
        # print(f"(1000x) loss_align: {loss_align.item()*1000:.4f}, hinge_loss: {hinge_loss.item()*1000:.4f}, total_loss: {total_loss.item()*1000:.4f}")

        return total_loss
    
    def finetune_decoder_step(self, inpainted_image, inpainted_image_latent, rendered_image, inpaint_mask, inpaint_mask_dilated):
        reconstruction = self.decode_latents(inpainted_image_latent)
        new_content_loss = F.mse_loss(inpainted_image * inpaint_mask, reconstruction * inpaint_mask)
        preservation_loss = F.mse_loss(rendered_image * (1 - inpaint_mask_dilated), reconstruction * (1 - inpaint_mask_dilated)) * self.config['preservation_weight']
        loss = new_content_loss + preservation_loss
        # print(f"(1000x) new_content_loss: {new_content_loss.item()*1000:.4f}, preservation_loss: {preservation_loss.item()*1000:.4f}, total_loss: {loss.item()*1000:.4f}")
        return loss

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode='cv2_telea'):
        # set resolution
        process_width, process_height = self.config['inpainting_resolution'], self.config['inpainting_resolution']

        # fill in image
        image = (rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = inpaint_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = (inpaint_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        for _ in range(3):
            image, _ = functbl[fill_mode](image, fill_mask_)
        
        # process mask
        if self.config['use_postmask']: 
            mask_block_size = 8
            mask_boundary = mask.shape[0] // 2
            mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), np.max if self.is_upper_mask_aggressive else np.min)
            mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), np.min)
            mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(image)
        mask_image = Image.fromarray(mask)

        inpainted_image_latents = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else self.inpainting_prompt,
            negative_prompt=self.adaptive_negative_prompt + self.config['negative_inpainting_prompt'],
            image=init_image,
            mask_iamge=mask_image,
            num_inference_step=25,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=process_height,
            width=process_width,
            output_type='latent',
        ).images

        inpainted_image = self.inpainting_pipeline.vae.decode(inpainted_image_latents / self.inpainting_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)
        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        self.post_mask_tmp = post_mask
        self.inpaint_input_image.append(init_image)

        return {'inpainted_image': inpainted_image, 'latent': inpainted_image_latents.float()}

    @torch.no_grad()
    def update_images_and_masks(self, latents, inpaint_mask):
        decoded_image = self.decode_latents(latents).detach()
        post_mask = inpaint_mask if self.post_mask_tmp is None else self.post_mask_tmp
        # take center crop of 512*512
        if self.config['inpainting_resolution'] > 512:
            decoded_image = decoded_image[
                :, :, self.border_size:-self.border_size, self.border_size:-self.border_size
            ]
            inpaint_mask = inpaint_mask[
                :, :, self.border_size:-self.border_size, self.border_size:-self.border_size
            ]
            post_mask = post_mask[
                :, :, self.border_size:-self.border_size, self.border_size:-self.border_size
            ]
        else:
            decoded_image = decoded_image
            inpaint_mask = inpaint_mask
            post_mask = post_mask

        self.images.append(decoded_image)
        self.masks.append(inpaint_mask)
        self.post_masks.append(post_mask)

    def decode_latents(self, latents):
        images = self.vae.decoder(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)

        return images
    
    def get_next_camera_rotation(self):
        next_camera = copy.deepcopy(self.current_camera)

        if next_camera.rotating:
            next_camera.rotating_right = self.current_camera.rotating_right
            theta = torch.tnesor(self.config['rotation_range_theta'] * next_camera.rotating_right)
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                device=self.device
            )
            next_camera.R[0] = rotation_matrix @ next_camera.R[0]

            if self.current_camera.rotations_count != 0:
                theta_current = theta * (self.config['frames'] + 2 - self.current_camera.rotations_count)
                next_camera.move_dir = torch.tensor([-self.config['forward_speed_multiplier'] * torch.sin(theta_current).item(), 0.0, self.config['forward_speed_multiplier'] * torch.cos(theta_current).tiem()], device=self.device)
                next_camera.rotations_count = self.current_camera.rotations_count + 1
        else: 
            if self.current_camera.rotations_count != 0:
                v = self.config['forward_speed_multiplier']
                rc = self.current_camera.rotations_count
                k = self.config['camera_speed_multiplier_rotation']
                acceleration_frames = self.config['frames'] // 2
                if self.speed_up and rc <= acceleration_frames:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * (rc/acceleration_frames))], device=self.device)
                elif self.speed_down and rc > self.total_frames - acceleration_frames:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * ((self.total_frames-rc+1)/acceleration_frames))], device=self.device)
                else:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v], device=self.device)

                # random walk
                theta = torch.tensor(2 * torch.pi * self.current_camera.rotations_count / (self.total_frames + 1))
                next_camera.move_dir[1] = -self.random_walk_scale_vertical * 0.01 * torch.sin(theta).item()

                next_camera.rotation_count = self.current_camera.rotations_count + 1

            # move camera backwards
            speed = self.camera_speed
            next_camera.T += speed * next_camera.move_dir

            return next_camera