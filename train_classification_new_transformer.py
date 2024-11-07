import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder, TransformerEncoderWithMask
from data import SentinelTimeSeriesDatasetForDownstreaming, SentinelTimeSeriesDatasetFixedTimestepForDownstreaming
from transforms import SampleValidPixels
import wandb
from einops import rearrange
import logging
import tqdm
from barlow_twins import BTModel, ProjectionHead


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        # self.ln1 = nn.LayerNorm(512)
        self.ln1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        # self.ln2 = nn.LayerNorm(256)
        self.ln2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        # self.ln3 = nn.LayerNorm(64)
        self.ln3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.relu1(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Layer 2
        x = self.relu2(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Layer 3
        x = self.relu3(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output layer (no activation here as we'll use softmax in the loss function)
        x = self.fc4(x)
        return x


# Backbone + Classification Head
class FullModel(nn.Module):
    def __init__(self, backbone, classification_head):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self.linear = nn.Linear(64, 11)

    def forward(self, x, mask):
        # x is (batch_size, time, bands)
        x = self.backbone(x, mask)
        # features = self.neck(features)
        # output = self.classification_head(features)
        output = self.linear(x)
        return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        if self.alpha is None:
            self.alpha = torch.tensor(1.0)

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# Define warmup scheduler (2000 steps to warmup)
def warmup_lambda(step):
    if step < 1000:
        return (step + 1) / 1000  # Linearly increase the learning rate
    else:
        return 1.0  # Maintain the learning rate

# Define constant learning rate for the middle phase (2000 to 12000 steps)
def constant_lambda(step):
    if step < 1000:
        return (step + 1) / 1000  # Warmup phase
    elif 1000 <= step < 12000:
        return 1.0  # Constant learning rate phase
    else:
        return 1.0  # After 12000, we will switch to CosineAnnealingLR

def train_model():
    # Initialize logging and wandb
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    wandb.init(project="btfm")

    run_config = {
    "backbone": "transformer",
    "backbone_param_hidden_dim": 128,
    "backbone_param_num_layers": 2,
    "min_valid": 32,
    "sample_size": 16,
    "band_size": 11,
    # "batch_size": 8192, # Maybe too large for BT, as BT doens't rely on contrastive learning.
    "batch_size": 2048,
    # "learning_rate": 0.00001,
    "learning_rate": 0.001,
    "epochs": 5,
    "latent_dim": 64,
    "validation_size": 1000,
    "warmup_steps": 1000,
    "warmdown_steps": 250000,
    "barlow_lambda": 5e-4,
    "projection_head_hidden_dim": 128,
    "projection_head_output_dim": 128,
    "train_dataset": "california",
    "val_dataset": "california",
    "time_dim": 0,
    "num_classes": 11
    }
    wandb.config.update(run_config)

    # Initialize dataset and data loaders
    dataset = SentinelTimeSeriesDatasetFixedTimestepForDownstreaming("../btfm-data-preparation/", "train_temp", run_config['min_valid'], buffer_size=600_000, shuffle=True)

    # test_dataset = SentinelTimeSeriesDatasetFixedTimestepForDownstreaming("../btfm-data-preparation/", "test", run_config['min_valid'], buffer_size=80_000, shuffle=True)
    test_dataset = dataset
    # debug: only take first tile
    # dataset.tiles = dataset.tiles[:1]
    # test_dataset.tiles = test_dataset.tiles[:1]
    
    train_dataloader = DataLoader(dataset, batch_size=run_config["batch_size"], num_workers=16,
                                                       pin_memory=True, 
                                                       drop_last=True, 
                                                       prefetch_factor=64
                                                       )
    test_dataloader = DataLoader(test_dataset, batch_size=run_config["batch_size"], num_workers=12,
                                                       pin_memory=True, 
                                                       persistent_workers=True,
                                                       drop_last=True, 
                                                       prefetch_factor=64
                                                       )

    logging.info(f"Train dataset size: {len(dataset)}, Test dataset size: {len(test_dataset)}")
    logging.info(f"Train dataloader size: {len(train_dataloader)}, Test dataloader size: {len(test_dataloader)}")

    # Load the backbone
    backbone = TransformerEncoderWithMask()
    
    # Load pre-trained weights
    checkpoint_path = "checkpoints/transformer_shape_96_11_latent_dim_64/model_checkpoint_val_best.pt"
    checkpoint = torch.load(checkpoint_path)
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Add neck
    # neck = SimpleAttentionNeck(run_config["latent_dim"])
    # Add classification head
    classification_head = ClassificationHead(run_config["latent_dim"], run_config["num_classes"])

    model = FullModel(backbone, classification_head).to('cuda')

    # 冻结所有的层
    for param in model.backbone.parameters():
        param.requires_grad = False

    # # 解冻最后的全连接层 (fc_out)
    # for param in model.backbone.fc_out.parameters():
    #     param.requires_grad = True

    # # 解冻最后一层 Transformer Encoder Layer
    # for param in model.backbone.transformer_encoder.layers[-1].parameters():
    #     param.requires_grad = True

    # Set optimizer and scheduler
    # 只更新解冻的层
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=run_config["learning_rate"])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=1e-6
    # )
    # Define the combined scheduler using SequentialLR
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,
    #     schedulers=[
    #         torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda),  # Warmup for first 2000 steps
    #         torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lambda),  # Keep constant from 2000 to 12000 steps
    #         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=0)  # Reduce to 0 gradually after 12000 steps
    #     ],
    #     milestones=[1000, 12000]
    # )
    
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    # criterion = FocalLoss(gamma=2, alpha=None, reduction='mean')
    criterion = nn.CrossEntropyLoss()

    step = 0

    # Training loop
    for epoch in range(run_config["epochs"]):
        model.train()
        total_loss = 0

        # for assembled_samples in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
        for i, batch_samples in enumerate(tqdm.tqdm(train_dataloader)):
            optimizer.zero_grad()
            sample = batch_samples[0].to('cuda', non_blocking=True)
            mask = batch_samples[1].to('cuda', non_blocking=True)
            label = batch_samples[2].to('cuda', non_blocking=True).long()

            outputs = model(sample, mask)
            loss = criterion(outputs, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 8 == 0:
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}, learning_rate: {current_learning_rate}")
                wandb.log({"epoch": epoch + 1, "loss": loss.item(), "learning_rate": current_learning_rate}, step=step)

            # 每隔X步进行验证
            # if step % 4096 == 0:
            # if step % 4096 == 0 and step != 0:
            #     model.eval()
            #     correct = 0
            #     total = 0
            #     total_val_loss = 0
            #     val_step = 0
            #     with torch.no_grad():
            #         for j, val_batch_samples in enumerate(tqdm.tqdm(test_dataloader)):
            #             val_sample = val_batch_samples[0].to('cuda', non_blocking=True)
            #             val_mask = val_batch_samples[1].to('cuda', non_blocking=True)
            #             val_label = val_batch_samples[2].to('cuda', non_blocking=True).long()
                        
            #             test_outputs = model(val_sample, val_mask)
            #             val_loss = criterion(test_outputs, val_label)
            #             total_val_loss += val_loss.item()
            #             probabilities = torch.softmax(test_outputs, dim=1)
            #             _, predicted = torch.max(probabilities, 1)
            #             total += val_label.size(0)
            #             correct += (predicted == val_label).sum().item()
            #             val_step += 1

            #             if val_step > run_config["validation_size"]:
            #                 break

            #     accuracy = 100 * correct / total
            #     avg_val_loss = total_val_loss / val_step
            #     logging.info(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
            #     wandb.log({"Validation_loss": avg_val_loss, "Validation_accuracy": accuracy}, step=step)
            #     model.train()
            
            step += 1

        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # Test
        # model.eval()
        # correct = 0
        # total = 0

        # with torch.no_grad():
        #     for assembled_samples in tqdm(test_dataloader, desc=f"Validating Epoch {epoch + 1}", unit="batch"):
        #         test_samples, doy_samples = assembled_samples
        #         test_samples = rearrange(test_samples, 'b p s n -> (b p) s n').to('cuda', non_blocking=True)
        #         doy_samples = rearrange(doy_samples, 'b p s -> (b p) s').to('cuda', non_blocking=True)
        #         test_inputs = test_samples[:, :, :-1]
        #         test_labels = test_samples[:, 0, -1].long()

        #         test_outputs = model(test_inputs,doy_samples)
        #         _, predicted = torch.max(test_outputs.data, 1)
        #         total += test_labels.size(0)
        #         correct += (predicted == test_labels).sum().item()

        # accuracy = 100 * correct / total
        # logging.info(f"Test Accuracy: {accuracy}%")
        # wandb.log({"Test_accuracy": accuracy}, step=step)

        # Save model every epoch
        # if (epoch + 1) % 1 == 0:
        #     model_path = f"checkpoints/{run_config['backbone']}_downstream_model_epoch_{epoch+1}.pth"
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #     }, model_path)
        #     logging.info(f"Model saved at epoch {epoch + 1} with accuracy {accuracy}%")

if __name__ == "__main__":
    train_model()