import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large



class TemporalConsistencyBase(nn.Module):


    def __init__(self, num_frames=8, raft_pretrained=True):
        super().__init__()
        self.num_frames = num_frames
        self.flow_model = self._initialize_raft(raft_pretrained)

    def _initialize_raft(self, pretrained=True):
        if pretrained:
            flow_model = raft_large(pretrained=True)
            for param in flow_model.parameters():
                param.requires_grad = False
        else:
            flow_model = raft_large(pretrained=False)
        flow_model.eval()
        return flow_model

    def estimate_optical_flow(self, frame1, frame2):

        with torch.no_grad():
            frame1_normalized = self._normalize_for_raft(frame1)
            frame2_normalized = self._normalize_for_raft(frame2)

            flow_predictions = self.flow_model(frame1_normalized, frame2_normalized)
            flow = flow_predictions[-1]

        return flow

    def _normalize_for_raft(self, frames):

        if frames.min() < 0:
            frames = (frames + 1) / 2
        elif frames.max() > 1:
            frames = frames / 255.0

        if frames.shape[1] == 1:
            frames = frames.repeat(1, 3, 1, 1)
        elif frames.shape[1] > 3:
            frames = frames[:, :3, :, :]

        return frames

    def warp_frame(self, frame, flow):

        B, C, H, W = frame.shape

        yy, xx = torch.meshgrid(
            torch.arange(H, device=frame.device),
            torch.arange(W, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

        grid = grid + flow.permute(0, 2, 3, 1)

        grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

        warped = F.grid_sample(
            frame,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped

    def compute_spatial_smoothness(self, frame):

        gradient_x = torch.abs(frame[:, :, :, 1:] - frame[:, :, :, :-1])
        gradient_y = torch.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :])

        smoothness_x = torch.mean(gradient_x, dim=1, keepdim=True)
        smoothness_y = torch.mean(gradient_y, dim=1, keepdim=True)

        smoothness_x = F.interpolate(smoothness_x, size=frame.shape[2:], mode='bilinear')
        smoothness_y = F.interpolate(smoothness_y, size=frame.shape[2:], mode='bilinear')

        smoothness = 1.0 / (1.0 + smoothness_x + smoothness_y + 1e-8)
        return smoothness

    def temporal_pool(self, temporal_features, scale):
        if scale == 1:
            return temporal_features

        B_T1, C, H, W = temporal_features.shape
        B = B_T1 // (self.num_frames - 1)

        seq_features = temporal_features.reshape(B, self.num_frames - 1, C, H, W)

        if seq_features.shape[1] >= scale:

            new_T = (self.num_frames - 1) // scale

            pooled_list = []
            for i in range(new_T):
                start_idx = i * scale
                end_idx = min((i + 1) * scale, self.num_frames - 1)
                segment = seq_features[:, start_idx:end_idx]  # [B, seg_len, C, H, W]
                segment_mean = torch.mean(segment, dim=1)  # [B, C, H, W]
                pooled_list.append(segment_mean)

            pooled_seq = torch.stack(pooled_list, dim=1)  # [B, new_T, C, H, W]

            return pooled_seq.reshape(-1, C, H, W)

        return temporal_features

    def perceptual_temporal_similarity(self, feat1, feat2):

        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)

        cosine_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=1)
        return torch.mean(1 - cosine_sim)

    def temporal_differences(self, frames):

        B_T, C, H, W = frames.shape
        frames_reshaped = frames.reshape(-1, self.num_frames, C, H, W)

        diffs = []
        for b in range(frames_reshaped.shape[0]):
            batch_diffs = []
            for t in range(self.num_frames - 1):
                diff = frames_reshaped[b, t + 1] - frames_reshaped[b, t]
                batch_diffs.append(diff)
            diffs.extend(batch_diffs)

        return torch.stack(diffs) if diffs else torch.tensor([], device=frames.device)




class ReconstructionTemporalLoss(TemporalConsistencyBase):

    def __init__(self, lambda_rec=1.0, num_frames=4):
        super().__init__(num_frames)
        self.lambda_rec = lambda_rec

    def forward(self, reconstructed, original):

        rec_diff = self.temporal_differences(reconstructed)
        orig_diff = self.temporal_differences(original)

        if rec_diff.numel() == 0 or orig_diff.numel() == 0:
            return torch.tensor(0.0, device=reconstructed.device)

        loss = 0
        scales = [1, 2, 4]
        valid_scales = 0

        for scale in scales:
            rec_scaled = self.temporal_pool(rec_diff, scale)
            orig_scaled = self.temporal_pool(orig_diff, scale)

            if rec_scaled.numel() > 0 and orig_scaled.numel() > 0:
                if rec_scaled.shape == orig_scaled.shape:
                    scale_loss = self.perceptual_temporal_similarity(rec_scaled, orig_scaled)
                    loss += scale_loss
                    valid_scales += 1

        return loss / max(valid_scales, 1) * self.lambda_rec


class GenerationTemporalLoss(TemporalConsistencyBase):


    def __init__(self, lambda_gen=1.0, num_frames=8, raft_pretrained=True):
        super().__init__(num_frames, raft_pretrained)
        self.lambda_gen = lambda_gen

    def forward(self, generated):

        B_T, C, H, W = generated.shape
        if B_T % self.num_frames != 0:
            actual_T = min(self.num_frames, B_T)
            seq = generated[:B_T // actual_T * actual_T].reshape(-1, actual_T, C, H, W)
        else:
            seq = generated.reshape(-1, self.num_frames, C, H, W)

        if seq.shape[1] < 2:
            return torch.tensor(0.0, device=generated.device)

        consistency_loss = 0
        T = seq.shape[1]

        flow_loss = 0
        flow_count = 0
        for t in range(T - 1):
            flow = self.estimate_optical_flow(seq[:, t], seq[:, t + 1])

            warped = self.warp_frame(seq[:, t], flow)

            frame_diff = torch.abs(warped - seq[:, t + 1])

            smoothness_weights = self.compute_spatial_smoothness(seq[:, t])
            weighted_diff = frame_diff * smoothness_weights

            flow_loss += torch.mean(weighted_diff)
            flow_count += 1

        if flow_count > 0:
            consistency_loss += flow_loss / flow_count

        if T >= 3:
            temp_smoothness = 0
            smooth_count = 0
            for t in range(1, T - 1):
                temp_accel = torch.abs(seq[:, t + 1] - 2 * seq[:, t] + seq[:, t - 1])

                temporal_variation = torch.abs(seq[:, t + 1] - seq[:, t - 1])
                acceleration_weight = 1.0 / (1.0 + temporal_variation + 1e-8)

                weighted_accel = temp_accel * acceleration_weight
                temp_smoothness += torch.mean(weighted_accel)
                smooth_count += 1

            if smooth_count > 0:
                consistency_loss += temp_smoothness / smooth_count


        return consistency_loss  * self.lambda_gen


