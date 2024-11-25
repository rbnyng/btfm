# training loop for barlow twins implementation
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from torch import nn
from torch.optim import AdamW
from data import SentinelTimeSeriesDataset, SentinelTimeSeriesDatasetFixedTimestep
# import intel_extension_for_pytorch as ipex
import wandb
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder, TransformerEncoderWithMask
from transforms import SampleValidPixels, DummyTransform
from barlow_twins import ProjectionHead, BarlowTwinsLoss, EncoderModel
from einops import rearrange
import time
import os
from datetime import datetime
import sys
import itertools
from utils import *
import subprocess
import argparse
from config import default_config
from torch.linalg import svd
import torch.nn.functional as F

class MatryoshkaTransformerWithMask(nn.Module):
    def __init__(self, input_dim=10, embed_dim=64, num_heads=8, hidden_dim=256, 
                 num_layers=6, nesting_dims=[32, 64, 128, 256, 512], dropout=0.2):
        super().__init__()
        self.nesting_dims = nesting_dims
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, 96, embed_dim))
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Matryoshka output layers for each dimension
        self.output_layers = nn.ModuleDict({
            f'dim_{dim}': nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, dim),
                nn.BatchNorm1d(dim)
            ) for dim in nesting_dims
        })
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = torch.clamp(x, min=-100, max=100)
        x = x + self.position_encoding
        
        if mask is not None:
            attention_mask = mask == 0
        else:
            attention_mask = None
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=attention_mask)
        
        weights = self.attention_weights(x)
        weights = torch.softmax(weights, dim=1)
        x = (x * weights).sum(dim=1)
        
        # Generate nested representations
        outputs = []
        for dim in self.nesting_dims:
            out = self.output_layers[f'dim_{dim}'](x)
            outputs.append(out)
            
        return outputs

class MatryoshkaCombinedLoss(nn.Module):
    def __init__(self, batch_size, barlow_lambda=5e-3, mmcr_alpha=5e-3, 
                 nesting_weights=None, matryoshka_lambda=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.barlow_lambda = barlow_lambda
        self.mmcr_alpha = mmcr_alpha
        self.matryoshka_lambda = matryoshka_lambda
        self.nesting_weights = nesting_weights or [1.0] * 5

    def off_diagonal_ele(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def compute_barlow_loss(self, z1, z2):
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size
        
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
        
        return on_diag + self.barlow_lambda * off_diag

    def compute_mmcr_loss(self, z1, z2):
        z_combined = torch.cat([z1, z2], dim=0)
        z_combined = z_combined - z_combined.mean(dim=0, keepdim=True)
        
        # Compute centroids and nuclear norm
        centroids = z_combined.view(self.batch_size, 2, -1).mean(dim=1)
        _, s_centroids, _ = torch.linalg.svd(centroids)
        mmcr_loss = -s_centroids.sum()
        
        # Sample expansion term
        _, s_z_combined, _ = torch.linalg.svd(z_combined)
        z_norm_loss = s_z_combined.sum() / self.batch_size
        
        return self.mmcr_alpha * (mmcr_loss + self.barlow_lambda * z_norm_loss)

    def compute_matryoshka_loss(self, z1s, z2s):
        # Ensure consistent representation across granularities
        consistency_loss = 0
        for i in range(len(z1s) - 1):
            # Compare adjacent granularities
            z1_curr = F.normalize(z1s[i], dim=1)
            z1_next = F.normalize(z1s[i + 1][:, :z1s[i].shape[1]], dim=1)
            z2_curr = F.normalize(z2s[i], dim=1)
            z2_next = F.normalize(z2s[i + 1][:, :z2s[i].shape[1]], dim=1)
            
            # Cosine similarity between representations
            consistency_loss += (2 - (F.cosine_similarity(z1_curr, z1_next).mean() +
                                    F.cosine_similarity(z2_curr, z2_next).mean()))
            
        return self.matryoshka_lambda * consistency_loss

    def forward(self, z1s, z2s):
        total_loss = 0
        barlow_losses = []
        mmcr_losses = []
        
        # Compute Barlow and MMCR losses for each granularity
        for z1, z2, weight in zip(z1s, z2s, self.nesting_weights):
            barlow_loss = self.compute_barlow_loss(z1, z2)
            mmcr_loss = self.compute_mmcr_loss(z1, z2)
            
            total_loss += weight * (barlow_loss + mmcr_loss)
            barlow_losses.append(barlow_loss.item())
            mmcr_losses.append(mmcr_loss.item())
        
        # Add Matryoshka consistency loss
        matryoshka_loss = self.compute_matryoshka_loss(z1s, z2s)
        total_loss += matryoshka_loss
        
        return total_loss, barlow_losses, mmcr_losses, matryoshka_loss.item()

def test_matryoshka_model_and_visualize(model, path_to_tile, test_batch_size=512, sample_size=16, n_pca_components=3, save_dir="."):
    # Load inference tile data
    tile_bands = np.load(path_to_tile + "/bands.npy")
    tile_masks = np.load(path_to_tile + "/masks.npy")
    tile_doys = np.load(path_to_tile + "/doys.npy")
    bands_mean = np.load(path_to_tile + "/band_mean.npy")
    bands_std = np.load(path_to_tile + "/band_std.npy")

    # Extract representations from encoder for each dimension
    representation_maps = extract_representations_fixed_timesteps(model, tile_bands, tile_masks, tile_doys, 
                                                               bands_mean, bands_std, test_batch_size)
    
    # Process each granularity level
    results = {}
    for dim_idx, rep_map in enumerate(representation_maps):
        dim = model.backbone.nesting_dims[dim_idx]
        
        # Save the representation map
        np.save(f"{save_dir}/{model.backbone.__class__.__name__}_dim_{dim}_representation_map.npy", rep_map)
        
        # Save visualization of channels
        for i in range(min(4, rep_map.shape[-1])):
            plt.imshow(rep_map[:, :, i])
            plt.title(f'Dim {dim} - Channel {i}')
            plt.axis('off')
            plt.imsave(f"{save_dir}/{model.backbone.__class__.__name__}_dim_{dim}_channel_{i}.png", 
                      rep_map[:, :, i])

        # PCA visualization for each dimension
        height, width = rep_map.shape[:2]
        flat_representation = rep_map.reshape(-1, rep_map.shape[-1])
        
        # Apply PCA
        pca = PCA(n_components=min(n_pca_components, rep_map.shape[-1]))
        pca_result = pca.fit_transform(flat_representation)
        
        # Reshape and process
        if pca_result.shape[1] >= 3:
            pca_image = pca_result.reshape(height, width, -1)[:, :, :3]
            
            # Scale channels to [0, 255]
            rgb_channels = []
            for c in range(3):
                channel = pca_image[:, :, c]
                channel = np.round((channel - np.min(channel)) / 
                                 (np.max(channel) - np.min(channel)) * 255)
                channel = np.clip(channel, 0, 255).astype(np.uint8)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channel = clahe.apply(channel)
                rgb_channels.append(channel)
            
            # Stack channels
            pca_image = np.stack(rgb_channels, axis=-1)
            
            # Save visualization
            plt.imshow(pca_image)
            plt.title(f'PCA Visualization - Dim {dim}')
            plt.axis('off')
            pca_image_path = f"{save_dir}/{model.backbone.__class__.__name__}_dim_{dim}_visualization.png"
            plt.imsave(pca_image_path, pca_image)
            
            # Log to wandb
            wandb.log({f"PCA Visualization - Dim {dim}": wandb.Image(pca_image_path)})
            
            results[dim] = pca_image
    
    return results
    
class MatryoshkaProjectionHead(nn.Module):
    def __init__(self, input_dims, hidden_dim=512, output_dims=None):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims or input_dims
        
        # Create separate projection heads for each granularity
        self.projectors = nn.ModuleDict({
            f'proj_{in_dim}': nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            ) for in_dim, out_dim in zip(input_dims, self.output_dims)
        })
    
    def forward(self, xs):
        # Project each granularity
        outputs = []
        for x, in_dim in zip(xs, self.input_dims):
            proj = self.projectors[f'proj_{in_dim}'](x)
            outputs.append(proj)
        return outputs
        
class MatryoshkaBTModel(nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
    
    def forward(self, x, mask=None):
        features = self.backbone(x, mask)
        projections = self.projector(features)
        return projections

def parse_args():
    parser = argparse.ArgumentParser(description="Barlow Twins Training Script")
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--device', type=str, help='Device to use for training (cuda or cpu)')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--project_name', type=str, help='WandB project name')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
    parser.add_argument('--backbone', type=str, help='Backbone model to use')
    parser.add_argument('--data_dir', type=str, help='Directory for SSL dataset storage')
    parser.add_argument('--train_dataset', type=str, help='Name of the training dataset')
    parser.add_argument('--val_dataset', type=str, help='Name of the validation dataset')
    parser.add_argument('--test_tile_path', type=str, help='Path to the test tile')
    parser.add_argument('--min_valid', type=int, help='Minimum number of valid timesteps in a pixel')
    parser.add_argument('--sample_size', type=int, help='Number of samples to take from each pixel')
    parser.add_argument('--band_size', type=int, help='Number of bands in the input data')
    # add more arguments here as needed
    return parser.parse_args()

def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:  # Only update config if a new value was provided
            config[key] = value
    return config

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load the default configuration and update it with command-line arguments
    run_config = update_config_with_args(default_config.copy(), args)

    # check that there are no modified files in the git repo
    # (that is, this run should be reproducible)
    # if there are modified files, we should not run the training

    modified_files = subprocess.run(["git", "status", "--porcelain", "-uno"], stdout=subprocess.PIPE).stdout.decode('utf-8')

    # if modified_files:
    #     logging.error("There are modified files in the git repo. Training should not be run. Commit or stash the changes and try again.")
    #     sys.exit(1)

    current_git_hash = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

    # create a folder to save weights
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_dir = f"checkpoints/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # debug TODO: remove
    # checkpoint_dir = "checkpoints/20241103_154214"

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    run = wandb.init(project='btfm')

    # update the run config with the git hash
    run_config["commit_link"] = f"https://gitlab.developers.cam.ac.uk/cst/eeg/btfm-training/-/commit/{current_git_hash}"

    wandb.config.update(run_config)

    dataset = SentinelTimeSeriesDatasetFixedTimestep(run_config['data_dir'], run_config['train_dataset'], run_config['min_valid'], sample_size=24, buffer_size=600_000, shuffle=True, is_training=True)

    # training_tiles_exclued = ["MGRS-12TYN", "MGRS-12TYR", "MGRS-12TVN", "MGRS-12TWN",
    #                         "MGRS-13TBF", "MGRS-13TCF", 'MGRS-13TDF', 'MGRS-13TDH', 'MGRS-13TEF']
    # dataset.tiles = [tile for tile in dataset.tiles if not any(excluded_tile in tile for excluded_tile in training_tiles_exclued)]
    # exclued the first tile
    dataset.tiles = dataset.tiles[1:]
    # debug
    # dataset.tiles = dataset.tiles[:1]
    # # debug
    # for sample in dataset:
    #     print(sample.shape)
    #     pass

    val_dataset = SentinelTimeSeriesDatasetFixedTimestep(run_config['data_dir'], run_config['train_dataset'], run_config['min_valid'], buffer_size=600_000, shuffle=False, is_training=False)

    val_dataset.tiles = val_dataset.tiles[:1]

    # img_interval_steps = val_interval_steps*4
    min_val_loss = 1e9
    val_best_model_path = os.path.join(checkpoint_dir, f'model_checkpoint_val_best.pt')

    logging.info(f"calculating number of samples in dataset")

    # this is a slow operation
    num_samples = len(dataset)

    wandb.config.update({"num_samples": num_samples})

    logging.info(f"num_samples: {num_samples}")

    # get validation set pixels and composite_rgb

    # val_composite_rgb, val_valid_pixels, val_valid_pixel_positions = get_tile(val_dataset)

    with logging_redirect_tqdm():
        logging.info(f"tiles: {len(dataset.tiles)}")

        # available_backbones = {"simple_mlp": SimpleMLP, "simple_cnn": SimpleCNN, "transformer": TransformerEncoder}

        # # extract the backbone params from the run_config, strip the backbone_param prefix
        # backbone_params = {k.replace("backbone_param_", ""): v for k, v in run_config.items() if "backbone_param_" in k}

        # backbone = available_backbones[run_config["backbone"]](run_config["sample_size"], run_config["band_size"], run_config["time_dim"], run_config["latent_dim"], **backbone_params)
        backbone = MatryoshkaTransformerWithMask(
            input_dim=10,
            embed_dim=64, 
            num_heads=8,
            hidden_dim=256,
            num_layers=6,
            nesting_dims=[32, 64, 128, 256, 512]
        )

        projection_heads = MatryoshkaProjectionHead(
            input_dims=[32, 64, 128, 256, 512],
            hidden_dim=512,
            output_dims=[512] * 5
        )

        model = MatryoshkaBTModel(backbone, projection_heads)

        # get number of parameters
        backbone_parameters = sum(p.numel() for p in backbone.parameters())
        projection_head_parameters = sum(p.numel() for p in projection_heads.parameters())
        total_parameters = backbone_parameters + projection_head_parameters

        logging.info(f"backbone parameters: {backbone_parameters}")
        logging.info(f"projection head parameters: {projection_head_parameters}")
        logging.info(f"total parameters: {total_parameters}")
        wandb.config.update({"backbone_parameters": backbone_parameters, "projection_head_parameters": projection_head_parameters, "total_parameters": total_parameters})

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=run_config["batch_size"], 
                                                       num_workers=12, 
                                                    #    num_workers=0, 
                                                       pin_memory=True, 
                                                       drop_last=True, 
                                                       prefetch_factor=16
                                                       )
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=run_config["batch_size"], 
                                                     num_workers=run_config["num_workers"], 
                                                     persistent_workers=True, 
                                                     pin_memory=True, 
                                                     prefetch_factor=16, 
                                                     drop_last=True)

        # barlow_loss = BarlowTwinsLoss(run_config["batch_size"], lambda_coeff=run_config["barlow_lambda"])

        criterion = MatryoshkaCombinedLoss(
            batch_size=run_config["batch_size"],
            barlow_lambda=run_config["barlow_lambda"],
            mmcr_alpha=run_config["mmcr_alpha"],
            nesting_weights=[0.5, 0.75, 1.0, 1.0, 1.0]
        )

        # we use the learning rate in the run_config for weights and 1/100th of that for biases (close to what the Barlow Twins paper used)
        weight_params = [p for n, p in model.named_parameters() if 'bias' not in n]
        bias_params = [p for n, p in model.named_parameters() if 'bias' in n]

        param_lrs = [
            {'params': weight_params, 'lr': run_config["learning_rate"]},
            {'params': bias_params, 'lr': run_config["learning_rate"] / 10}
        ]

        optimizer = AdamW(param_lrs, weight_decay=1.5e-6)
        model = model.to('cuda')
        # model = torch.compile(model, fullgraph=True)

        total_steps = run_config["epochs"] * num_samples / run_config["batch_size"]

        # create scheduler that warms up for warmup steps up to learning rate
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1) / run_config["warmup_steps"], 1.0))
        # warm up for 1000 steps, do not use the lambda function
        # scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, run_config["learning_rate"]/10)
        # create a scheduler that is constant until the last warmdown steps
        scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, run_config["learning_rate"])
        # create a scheduler that decays the learning rate to 0 at the end of training
        scheduler3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=run_config["warmdown_steps"])

        # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[run_config["warmup_steps"], total_steps-run_config["warmdown_steps"]])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[run_config["warmup_steps"]])
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 0.01)
        step = 0
        examples = 0
        last_examples = 0
        last_update_time = time.time()
        running_loss = []
        running_loss_steps = 40 # in 256 steps

        # val_iterator = iter(val_dataloader)

        wandb.watch(model, log="all")

        for epoch in range(run_config["epochs"]):
            for i, batch_samples in enumerate(tqdm.tqdm(train_dataloader)):
                optimizer.zero_grad()
                sample0 = batch_samples[0].to('cuda', non_blocking=True)
                mask0 = batch_samples[1].to('cuda', non_blocking=True)
                sample1 = batch_samples[2].to('cuda', non_blocking=True)
                mask1 = batch_samples[3].to('cuda', non_blocking=True)
                
                z0 = model(sample0, mask0)
                z1 = model(sample1, mask1)

                loss, barlowloss, mmcrloss, matryoshka_loss = criterion(z0, z1)
                loss.backward()
                
                # clamp the gradient to prevent large gradients from dodgy batches
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()

                scheduler.step()

                examples += batch_samples[0].shape[0]

                if step % run_config["log_interval_steps"] == 0:
                    current_time = time.time()
                    
                    try:
                        effective_rank = rankme(z0)
                    except torch._C._LinAlgError as e:
                        print("Error during SVD computation:", e)
                        print("Printing z0 for debugging:")
                        print(z0)
                        torch.save(z0, 'z0_debug.pt') 
                        sys.exit(1)

                    examples_per_sec = (examples-last_examples) / (current_time - last_update_time)
                    current_learning_rate = optimizer.param_groups[0]['lr']
                    if len(running_loss) == running_loss_steps:
                        running_loss.pop(0)
                    running_loss.append(loss.item())
                    running_loss_avg = sum(running_loss) / len(running_loss)
                    batch_size = batch_samples[0].shape[0]
                    batch_std = batch_samples[0][:,0].std()
                    
                    
                    # Format losses for logging
                    barlow_loss_str = ", ".join([f"barlow_{i}: {loss:.2f}" for i, loss in enumerate(barlowloss)])
                    mmcr_loss_str = ", ".join([f"mmcr_{i}: {loss:.2f}" for i, loss in enumerate(mmcrloss)])
                    rank_str = ", ".join([f"rank_{i}: {rank:.2f}" for i, rank in enumerate(effective_rank)])
            
                    # logging.info(f"step: {step}, epoch: {epoch}, loss: {loss.item():.2f}, r. loss: {running_loss_avg:.2f}, examples/sec: {examples_per_sec:.2f}, lr: {current_learning_rate}, barlowloss: {barlowloss.item():.2f}, mmcrloss: {mmcrloss.item():.2f}, matryoshkaloss: {matryoshka_loss.item():.2f}, effective_rank: {effective_rank:.2f}, batch_size: {batch_size}, batch_std: {batch_std:.2f}")
                    # wandb.log({"epoch": epoch, "effective_rank": effective_rank, "loss": loss.item(), "barlowloss": barlowloss.item(), "mmcrloss": mmcrloss.item(), "matryoshkaloss": matryoshka_loss.item(), "examples_per_sec": examples_per_sec, "examples": examples, "lr": current_learning_rate, "batch_size": batch_size, "batch_std": batch_std}, step=step)
                    
                    
                    logging.info(
                        f"step: {step}, "
                        f"epoch: {epoch}, "
                        f"loss: {loss.item():.2f}, "
                        f"r. loss: {running_loss_avg:.2f}, "
                        f"examples/sec: {examples_per_sec:.2f}, "
                        f"lr: {current_learning_rate}, "
                        f"{barlow_loss_str}, "
                        f"{mmcr_loss_str}, "
                        f"matryoshka_loss: {matryoshka_loss:.2f}, "
                        f"{rank_str}, "
                        f"batch_size: {batch_size}, "
                        f"batch_std: {batch_std:.2f}"
                    )
                    
                    
                    # Log to wandb
                    metrics = {
                        "epoch": epoch,
                        "total_loss": loss.item(),
                        "running_loss": running_loss_avg,
                        "examples_per_sec": examples_per_sec,
                        "learning_rate": current_learning_rate,
                        "matryoshka_loss": matryoshka_loss,
                        "batch_size": batch_size,
                        "batch_std": batch_std.item()
                    }
                    
                    # Add individual losses and ranks for each dimension
                    for i, (barlow_loss, mmcr_loss, rank) in enumerate(zip(barlowloss, mmcrloss, effective_rank)):
                        dim = model.backbone.nesting_dims[i]
                        metrics.update({
                            f"barlow_loss_dim_{dim}": barlow_loss,
                            f"mmcr_loss_dim_{dim}": mmcr_loss,
                            f"effective_rank_dim_{dim}": rank
                        })
                    
                    wandb.log(metrics, step=step)
            

                    last_update_time = current_time
                    last_examples = examples

                if step % 10000 == 0 and step > 0: # save model every 10,000 steps
                # if step % 10000 == 0:
                    model_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{step}.pt')
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)

                # if step % run_config["val_interval_steps"] == 0 and step > 0:
                if step % run_config["val_interval_steps"] == 0:
                    model.eval()  # set model to evaluation mode
                    val_losses = {
                        'total_loss': 0,
                        'matryoshka_loss': 0
                    }
                    # Initialize lists for storing dimension-specific losses
                    val_barlow_losses = [0] * len(model.backbone.nesting_dims)
                    val_mmcr_losses = [0] * len(model.backbone.nesting_dims)
                    val_effective_ranks = [0] * len(model.backbone.nesting_dims)
                    
                    val_steps = run_config["validation_size"] // run_config["batch_size"]
                    
                    with torch.no_grad():  # disable gradient calculation
                        for k, val_batch_samples in enumerate(tqdm.tqdm(val_dataloader, total=val_steps)):
                            val_sample0 = val_batch_samples[0].to('cuda', non_blocking=True)
                            val_mask0 = val_batch_samples[1].to('cuda', non_blocking=True)
                            val_sample1 = val_batch_samples[2].to('cuda', non_blocking=True)
                            val_mask1 = val_batch_samples[3].to('cuda', non_blocking=True)
                            
                            z_val0 = model(val_sample0, val_mask0)
                            z_val1 = model(val_sample1, val_mask1)
                            
                            loss_val, batch_barlow_losses, batch_mmcr_losses, batch_matryoshka_loss = criterion(z_val0, z_val1)
                            
                            # Accumulate total loss and matryoshka loss
                            val_losses['total_loss'] += loss_val.item()
                            val_losses['matryoshka_loss'] += batch_matryoshka_loss
                            
                            # Accumulate dimension-specific losses
                            for i in range(len(val_barlow_losses)):
                                val_barlow_losses[i] += batch_barlow_losses[i]
                                val_mmcr_losses[i] += batch_mmcr_losses[i]
                                val_effective_ranks[i] += rankme(z_val0[i])  # Calculate rank for each dimension
                            
                            if k == val_steps:
                                break
                    
                    # Average the losses
                    val_steps = min(val_steps, k + 1)
                    for key in val_losses:
                        val_losses[key] /= val_steps
                        
                    for i in range(len(val_barlow_losses)):
                        val_barlow_losses[i] /= val_steps
                        val_mmcr_losses[i] /= val_steps
                        val_effective_ranks[i] /= val_steps
                    
                    # Log validation metrics
                    logging.info(
                        f"step {step}, epoch: {epoch}, "
                        f"validation loss: {val_losses['total_loss']:.4f}, "
                        f"matryoshka loss: {val_losses['matryoshka_loss']:.4f}"
                    )
                    
                    # Prepare metrics for wandb logging
                    metrics = {
                        "validation_loss": val_losses['total_loss'],
                        "val_matryoshka_loss": val_losses['matryoshka_loss']
                    }
                    
                    # Add dimension-specific metrics
                    for i, dim in enumerate(model.backbone.nesting_dims):
                        metrics.update({
                            f"val_barlow_loss_dim_{dim}": val_barlow_losses[i],
                            f"val_mmcr_loss_dim_{dim}": val_mmcr_losses[i],
                            f"val_effective_rank_dim_{dim}": val_effective_ranks[i]
                        })
                    
                    wandb.log(metrics, step=step)
                    
                    # Plot cross-correlation matrix for the largest dimension
                    cross_corr_plot = plot_cross_corr(z_val0[-1], z_val1[-1])  # Use the largest dimension
                    wandb.log({"cross_correlation": wandb.Image(cross_corr_plot)}, step=step)
                    
                    # Save best model based on total validation loss
                    if val_losses['total_loss'] < min_val_loss:
                        min_val_loss = val_losses['total_loss']
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, val_best_model_path)
                    
                    model.train()  # reset model to training mode

                step += 1
                
                # debug 
                # if step >= 1010:
                #     break

            model.train()  # reset model to training mode

        # model_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{step}.pt')
        # torch.save({
        #     'step': step,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, model_path)

        # add wandb artifact
        # artifact = wandb.Artifact('model', type='model')
        # artifact.add_file(model_path)
        # run.log_artifact(artifact)

        # # add the best validation model as an artifact
        # artifact = wandb.Artifact('best_model', type='model')
        # artifact.add_file(val_best_model_path)
        # run.log_artifact(artifact)
        
        # test the best model and save the false color map
        ############################
        checkpoint = torch.load(val_best_model_path, weights_only=True)
        backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Load the trained encoder model
        test_model = EncoderModel(backbone, run_config["time_dim"])
        # set to evaluation mode
        test_model.eval()
        logging.info(f"Testing the best model on the test tile")
        test_matryoshka_model_and_visualize(test_model, run_config["test_tile_path"], test_batch_size=512, sample_size=run_config["sample_size"], save_dir=checkpoint_dir)
        ############################

        run.finish()
        
if __name__ == "__main__":
    main()