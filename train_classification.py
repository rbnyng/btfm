import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder
from data import SentinelTimeSeriesDatasetForDownstreaming
from transforms import SampleValidPixels
import wandb
from einops import rearrange
import logging
import tqdm
from barlow_twins import BTModel, ProjectionHead


class SimpleAttentionNeck(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttentionNeck, self).__init__()
        # Attention weights for each feature dimension, so the output will still be (batch_size, input_dim)
        self.attention_weights = nn.Linear(input_dim, input_dim)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights for each feature
        weights = self.sigmoid(self.attention_weights(x))  # Shape: (batch_size, input_dim)
        return x * weights  # Element-wise multiplication with the attention weights


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Layer 2
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer (no activation here as we'll use softmax in the loss function)
        x = self.fc3(x)
        return x


# Backbone + Classification Head
class FullModel(nn.Module):
    def __init__(self, backbone, neck, classification_head, time_dimension):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.classification_head = classification_head
        self.time_dim = time_dimension
        # 如果为0，那么不需要time_embedding，否则为None
        self.time_embedding = None if time_dimension == 0 else nn.Embedding(53, time_dimension)
        self.linear = nn.Linear(64, 11)

    def forward(self, x, time):
        # x is (batch_size, time, bands)
        # time is (batch_size, time)
        # we embed the time and then concatenate it with the bands
        if self.time_dim != 0:
            time = self.time_embedding(time)
            x = torch.cat([x, time], dim=-1)
        features = self.backbone(x)
        # features = self.neck(features)
        # output = self.classification_head(features)
        output = self.linear(features)
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
    if step < 2000:
        return (step + 1) / 2000  # Linearly increase the learning rate
    else:
        return 1.0  # Maintain the learning rate

# Define constant learning rate for the middle phase (2000 to 12000 steps)
def constant_lambda(step):
    if step < 2000:
        return (step + 1) / 2000  # Warmup phase
    elif 2000 <= step < 12000:
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
    "batch_size": 256,
    # "learning_rate": 0.00001,
    "learning_rate": 0.0001,
    "epochs": 5,
    "latent_dim": 64,
    "validation_size": 10000,
    "warmup_steps": 100,
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
    transform = SampleValidPixels(run_config["sample_size"])
    dataset = SentinelTimeSeriesDatasetForDownstreaming('train', run_config["sample_size"], transform, shuffle=True)
    test_dataset = SentinelTimeSeriesDatasetForDownstreaming('test', run_config["sample_size"], transform, shuffle=False)
    
    # debug: only take first tile
    # dataset.tiles = dataset.tiles[:1]
    # test_dataset.tiles = test_dataset.tiles[:1]
    
    train_dataloader = DataLoader(dataset, batch_size=run_config["batch_size"], num_workers=7)
    test_dataloader = DataLoader(test_dataset, batch_size=run_config["batch_size"], num_workers=2)

    logging.info(f"Train dataset size: {len(train_dataloader)}, Test dataset size: {len(test_dataset)}")

    # Load the backbone
    available_backbones = {"simple_mlp": SimpleMLP, "simple_cnn": SimpleCNN, "transformer": TransformerEncoder}
    # extract the backbone params from the run_config, strip the backbone_param prefix
    backbone_params = {k.replace("backbone_param_", ""): v for k, v in run_config.items() if "backbone_param_" in k}

    backbone = available_backbones[run_config["backbone"]](run_config["sample_size"], run_config["band_size"], run_config["time_dim"], run_config["latent_dim"], **backbone_params)

    # projection_head = ProjectionHead(run_config["latent_dim"], run_config["projection_head_hidden_dim"], run_config["projection_head_output_dim"])

    # model = BTModel(backbone, projection_head, run_config["time_dim"])

    # Load pre-trained weights
    checkpoint_path = "checkpoints/20241004_170541/model_checkpoint_val_best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Add neck
    neck = SimpleAttentionNeck(run_config["latent_dim"])
    # Add classification head
    classification_head = ClassificationHead(run_config["latent_dim"], run_config["num_classes"])

    model = FullModel(backbone, neck, classification_head, run_config["time_dim"]).to('cuda')

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
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda),  # Warmup for first 2000 steps
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lambda),  # Keep constant from 2000 to 12000 steps
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=0)  # Reduce to 0 gradually after 12000 steps
        ],
        milestones=[2000, 13000]
    )

    # criterion = FocalLoss(gamma=2, alpha=None, reduction='mean')
    criterion = nn.CrossEntropyLoss()

    step = 0

    # Training loop
    for epoch in range(run_config["epochs"]):
        model.train()
        total_loss = 0

        # for assembled_samples in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
        for i, assembled_samples in enumerate(tqdm.tqdm(train_dataloader)):
            batch_samples, doy_samples = assembled_samples
            # batch_samples = batch_samples.to('cuda', non_blocking=True)

            batch_samples = rearrange(batch_samples, 'b p s n -> (b p) s n').to('cuda', non_blocking=True)
            doy_samples = rearrange(doy_samples, 'b p s -> (b p) s').to('cuda', non_blocking=True)
            input1 = batch_samples[:, :, :-1]
            labels = batch_samples[:, 0, -1].long()

            outputs = model(input1, doy_samples)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 128 == 0:
                current_learning_rate = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}, learning_rate: {current_learning_rate}")
                wandb.log({"epoch": epoch + 1, "loss": loss.item(), "learning_rate": current_learning_rate}, step=step)

            # 每隔X步进行验证
            # if step % 4096 == 0:
            if step % 4096 == 0 and step != 0:
                model.eval()
                correct = 0
                total = 0
                total_val_loss = 0
                val_step = 0
                with torch.no_grad():
                    for j, assembled_samples in enumerate(tqdm.tqdm(test_dataloader)):
                        test_samples, doy_samples = assembled_samples
                        test_samples = rearrange(test_samples, 'b p s n -> (b p) s n').to('cuda', non_blocking=True)
                        doy_samples = rearrange(doy_samples, 'b p s -> (b p) s').to('cuda', non_blocking=True)
                        test_inputs = test_samples[:, :, :-1]
                        test_labels = test_samples[:, 0, -1].long()

                        test_outputs = model(test_inputs, doy_samples)
                        val_loss = criterion(test_outputs, test_labels)
                        total_val_loss += val_loss.item()
                        _, predicted = torch.max(test_outputs.data, 1)
                        total += test_labels.size(0)
                        correct += (predicted == test_labels).sum().item()
                        val_step += 1

                        if val_step > run_config["validation_size"]:
                            break

                accuracy = 100 * correct / total
                avg_val_loss = total_val_loss / val_step
                logging.info(f"Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
                wandb.log({"Validation_loss": avg_val_loss, "Validation_accuracy": accuracy}, step=step)
                model.train()
            
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
        if (epoch + 1) % 1 == 0:
            model_path = f"checkpoints/{run_config['backbone']}_downstream_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)
            logging.info(f"Model saved at epoch {epoch + 1} with accuracy {accuracy}%")

if __name__ == "__main__":
    train_model()