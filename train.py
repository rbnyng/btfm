# training loop for barlow twins implementation
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch
from torch import nn
from torch.optim import AdamW
from data import SentinelTimeSeriesDataset
# import intel_extension_for_pytorch as ipex
import wandb
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder
from transforms import SampleValidPixels, DummyTransform
from barlow_twins import ProjectionHead, BarlowTwinsLoss, BTModel, EncoderModel
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

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    run = wandb.init(project='btfm')

    # update the run config with the git hash
    run_config["commit_link"] = f"https://gitlab.developers.cam.ac.uk/cst/eeg/btfm-training/-/commit/{current_git_hash}"

    wandb.config.update(run_config)

    transform = SampleValidPixels(run_config["sample_size"], seed=42, sample_band=True)
    dataset = SentinelTimeSeriesDataset(run_config["data_dir"], run_config["train_dataset"], run_config["min_valid"], transform, shuffle=True)

    # training_tiles_exclued = ["MGRS-12TYN", "MGRS-12TYR", "MGRS-12TVN", "MGRS-12TWN",
    #                         "MGRS-13TBF", "MGRS-13TCF", 'MGRS-13TDF', 'MGRS-13TDH', 'MGRS-13TEF']
    # dataset.tiles = [tile for tile in dataset.tiles if not any(excluded_tile in tile for excluded_tile in training_tiles_exclued)]
    # exclued the first tile
    # dataset.tiles = dataset.tiles[1:]
    dataset.tiles = dataset.tiles[:1]
    # # debug
    # for sample in dataset:
    #     print(sample.shape)
    #     pass

    val_transform = SampleValidPixels(run_config["sample_size"], seed=42, sample_band=False)
    val_dataset = SentinelTimeSeriesDataset(run_config["data_dir"], run_config["val_dataset"], run_config["min_valid"], val_transform, shuffle=False, 
                                            buffer_size=run_config["validation_size"], shuffle_worker_id=False)

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

        available_backbones = {"simple_mlp": SimpleMLP, "simple_cnn": SimpleCNN, "transformer": TransformerEncoder}

        # extract the backbone params from the run_config, strip the backbone_param prefix
        backbone_params = {k.replace("backbone_param_", ""): v for k, v in run_config.items() if "backbone_param_" in k}

        backbone = available_backbones[run_config["backbone"]](run_config["sample_size"], run_config["band_size"], run_config["time_dim"], run_config["latent_dim"], **backbone_params)

        projection_head = ProjectionHead(run_config["latent_dim"], run_config["projection_head_hidden_dim"], run_config["projection_head_output_dim"])

        model = BTModel(backbone, projection_head)

        # get number of parameters
        backbone_parameters = sum(p.numel() for p in backbone.parameters())
        projection_head_parameters = sum(p.numel() for p in projection_head.parameters())
        total_parameters = backbone_parameters + projection_head_parameters

        logging.info(f"backbone parameters: {backbone_parameters}")
        logging.info(f"projection head parameters: {projection_head_parameters}")
        logging.info(f"total parameters: {total_parameters}")
        wandb.config.update({"backbone_parameters": backbone_parameters, "projection_head_parameters": projection_head_parameters, "total_parameters": total_parameters})

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=run_config["batch_size"], num_workers=run_config["num_workers"], 
                                                       pin_memory=True, drop_last=True, prefetch_factor=16)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=run_config["batch_size"], num_workers=1, persistent_workers=True, 
                                                     pin_memory=True, prefetch_factor=64, drop_last=True)

        barlow_loss = BarlowTwinsLoss(run_config["batch_size"], lambda_coeff=run_config["barlow_lambda"])

        # we use the learning rate in the run_config for weights and 1/100th of that for biases (close to what the Barlow Twins paper used)
        weight_params = [p for n, p in model.named_parameters() if 'bias' not in n]
        bias_params = [p for n, p in model.named_parameters() if 'bias' in n]

        param_lrs = [
            {'params': weight_params, 'lr': run_config["learning_rate"]},
            {'params': bias_params, 'lr': run_config["learning_rate"] / 10}
        ]

        optimizer = AdamW(param_lrs, weight_decay=1.5e-6)
        model = model.to('cuda')
        model = torch.compile(model, fullgraph=True)

        total_steps = run_config["epochs"] * num_samples / run_config["batch_size"]

        # create scheduler that warms up for warmup steps up to learning rate
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1) / run_config["warmup_steps"], 1.0))
        # create a scheduler that is constant until the last warmdown steps
        scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer, run_config["learning_rate"])
        # create a scheduler that decays the learning rate to 0 at the end of training
        scheduler3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=run_config["warmdown_steps"])

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[run_config["warmup_steps"], total_steps-run_config["warmdown_steps"]])
        # scheduler = scheduler2

        step = 0
        examples = 0
        last_examples = 0
        last_update_time = time.time()
        running_loss = []
        running_loss_steps = 40 # in 256 steps

        val_iterator = iter(val_dataloader)

        wandb.watch(model, log="all")

        for epoch in range(run_config["epochs"]):
            for i, batch_doy_samples in enumerate(tqdm.tqdm(train_dataloader)):
                optimizer.zero_grad()
                batch_samples, doy_samples = batch_doy_samples
                batch_samples = batch_samples.to('cuda', non_blocking=True) # torch.Size([256, 2, 16, 11])
                doy_samples = doy_samples.to('cuda', non_blocking=True) # torch.Size([256, 2, 16])
                
                # test Keshav's thoughts
                # populate the input tensor with 1
                # batch_samples = torch.ones_like(batch_samples).to('cuda')
                # doy_samples = torch.ones_like(doy_samples).to('cuda')
                
                z0 = model(batch_samples[:,0].squeeze(), doy_samples[:,0].squeeze())
                z1 = model(batch_samples[:,1].squeeze(), doy_samples[:,1].squeeze())
                loss, on_diag, off_diag = barlow_loss(z0, z1)
                loss.backward()
                # clamp the gradient to prevent large gradients from dodgy batches
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()

                scheduler.step()

                examples += batch_samples.shape[0]

                if i % run_config["log_interval_steps"] == 0:
                    current_time = time.time()
                    effective_rank = rankme(z0)
                    examples_per_sec = (examples-last_examples) / (current_time - last_update_time)
                    current_learning_rate = optimizer.param_groups[0]['lr']
                    if len(running_loss) == running_loss_steps:
                        running_loss.pop(0)
                    running_loss.append(loss.item())
                    running_loss_avg = sum(running_loss) / len(running_loss)
                    batch_size = batch_samples.shape[0]
                    batch_std = batch_samples[:,0].std()
                    logging.info(f"step: {step}, epoch: {epoch}, loss: {loss.item():.2f}, r. loss: {running_loss_avg:.2f}, examples/sec: {examples_per_sec:.2f}, lr: {current_learning_rate}, on_diag: {on_diag.item():.2f}, off_diag: {off_diag.item():.2f}, effective_rank: {effective_rank:.2f}, batch_size: {batch_size}, batch_std: {batch_std:.2f}")
                    wandb.log({"epoch": epoch, "effective_rank": effective_rank, "loss": loss.item(), "on_diag": on_diag.item(), "off_diag": off_diag.item(), "examples_per_sec": examples_per_sec, "examples": examples, "lr": current_learning_rate, "batch_size": batch_size, "batch_std": batch_std}, step=step)

                    last_update_time = current_time
                    last_examples = examples

                if i % 100000 == 0 and step > 0: # save model every 10,000 steps
                    model_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{step}.pt')
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)

                if i % run_config["val_interval_steps"] == 0:
                    model.eval()  # set model to evaluation mode
                    val_loss = 0
                    val_on_diag_loss = 0
                    val_off_diag_loss = 0
                    val_effective_rank = 0
                    val_steps = run_config["validation_size"] // run_config["batch_size"]
                    with torch.no_grad():  # disable gradient calculation
                        for i, val_doy_samples in enumerate(tqdm.tqdm(val_iterator, total=val_steps)):
                            val_samples, val_doy_samples = val_doy_samples
                            val_samples = val_samples.to('cuda', non_blocking=True)
                            val_doy_samples = val_doy_samples.to('cuda', non_blocking=True)
                            z_val0 = model(val_samples[:,0].squeeze(), val_doy_samples[:,0].squeeze())
                            z_val1 = model(val_samples[:,1].squeeze(), val_doy_samples[:,1].squeeze())
                            loss_val, val_on_diag, val_off_diag = barlow_loss(z_val0, z_val1)
                            val_loss += loss_val.item()
                            val_on_diag_loss += val_on_diag.item()
                            val_off_diag_loss += val_off_diag.item()
                            val_effective_rank += rankme(z_val0)
                            if i == val_steps:
                                break
                    val_iterator = iter(val_dataloader)
                    val_loss /= val_steps
                    val_on_diag_loss /= val_steps
                    val_off_diag_loss /= val_steps
                    val_effective_rank /= val_steps
                    logging.info(f"step {step}, epoch: {epoch}, validation loss: {val_loss}, on_diag_loss: {val_on_diag_loss}, off_diag_loss: {val_off_diag_loss}, val_effective_rank: {val_effective_rank}")
                    wandb.log({"validation_loss": val_loss, "val_effective_rank": val_effective_rank, "val_on_diag_loss": val_on_diag_loss, "val_off_diag_loss": val_off_diag_loss}, step=step)

                    cross_corr_plot = plot_cross_corr(z_val0, z_val1)
                    wandb.log({"cross_correlation": wandb.Image(cross_corr_plot)}, step=step)

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
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

        model_path = os.path.join(checkpoint_dir, f'model_checkpoint_step_{step}.pt')
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)

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
        # checkpoint = torch.load(val_best_model_path, weights_only=True)
        # backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # # Load the trained encoder model
        # test_model = EncoderModel(backbone, run_config["time_dim"])
        # # set to evaluation mode
        # test_model.eval()
        # test_model_and_visualize(test_model, run_config["test_tile_path"], test_batch_size=512, sample_size=run_config["sample_size"], save_dir=checkpoint_dir)
        ############################

        run.finish()
        
if __name__ == "__main__":
    main()