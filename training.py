import os
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from ptflops import get_model_complexity_info
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from data_loader import Crack_loader
from models.model import TransMUNet
from utils.utils import setup_seed, Visualizer
from utils.loss import DiceBCELoss
from utils.metrics import IoU, Dice_Coeff

setup_seed(42)
torch.cuda.empty_cache()


class TrainPipeline:
    def __init__(self, config):
        self.config = config
        self.console = Console()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TransMUNet(n_classes=1).to(self.device)
        self.citerion = DiceBCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config['lr']))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=config['patience']
        )
        self.visualizer = Visualizer(isTrain=True)

        self.best_val_loss = np.inf
        self.best_epoch = 0

        self.visualizer = Visualizer(isTrain=True)
        self.best_val_loss=np.inf
        self.best_epoch=0
        self.prepare_data()
        self.load_pretrained()
        self.compute_flops()
    def prepare_data(self):
        DIR_IMAGE_TRAIN = os.path.join(self.config['path_to_tradata'], 'images')
        DIR_MASK_TRAIN = os.path.join(self.config['path_to_tradata'], 'masks')
        DIR_IMAGE_VAL = os.path.join(self.config['path_to_valdata'], 'images')
        DIR_MASK_VAL = os.path.join(self.config['path_to_valdata'], 'masks')

        img_names_train = [path.name for path in Path(DIR_IMAGE_TRAIN).glob('*.jpg')]
        mask_names_train = [path.name for path in Path(DIR_MASK_TRAIN).glob('*.png')]
        img_names_val = [path.name for path in Path(DIR_IMAGE_VAL).glob('*.jpg')]
        mask_names_val = [path.name for path in Path(DIR_IMAGE_VAL).glob('*.png')]

        train_dataset = Crack_loader(DIR_IMAGE_TRAIN, img_names_train, DIR_MASK_TRAIN, mask_names_train, isTrain=True)
        valid_dataset = Crack_loader(DIR_IMAGE_VAL, img_names_val, DIR_MASK_VAL, mask_names_val, resize=True)

        self.train_loader = DataLoader(train_dataset, batch_size=int(self.config['batch_size_tr']), shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=int(self.config['batch_size_va']), shuffle=False, drop_last=True, num_workers=2, pin_memory=True)

    def load_pretrained(self):
        if self.config['pretrained']:
            checkpoint = torch.load(self.config['save_model'], map_location='cpu', weights_only=True)
            self.model.load_state_dict(checkpoint['model_weights'], strict=False)
            self.best_val_loss = checkpoint['val_loss']
    
    def compute_flops(self):
        flops, params = get_model_complexity_info(self.model, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
        message = f'flops: {flops}, params: {params}'
        print(message)
        log_path = os.path.join('./checkpoints', self.config['loss_filename'])
        with open(log_path, "a") as log_file:
            log_file.write(f"{message}\n")

    def train_one_epochs(self, epoch):
        self.model.train()
        train_loss, iou_score, dice_score = 0, [], []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as progress:
            task = progress.add_task("[green] Training...", total=len(self.train_loader))
            
            for batch in self.train_loader:
                img = batch['image'].float().to(self.device)
                mask = batch['mask'].float().to(self.device)
                boundary = batch['boundary'].float().to(self.device)

                mask_pred, boundary_pred = self.model(img, isTrain=True)
                mask_pred = torch.sigmoid(mask_pred)

                loss_main = self.citerion(mask_pred, mask)
                loss_bnd = self.citerion(boundary_pred, boundary)
                loss = 0.8*loss_main + 0.2*loss_bnd

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                

                iou_score.append(IoU()(mask_pred, mask).item())
                dice_score.append(Dice_Coeff()(mask_pred, mask).item())

                progress.advance(task)
                mean_train_loss = train_loss/len(self.train_loader)
                mean_iou_score = np.mean(iou_score) * 100
                mean_dice_score = np.mean(dice_score) * 100
                
        return mean_train_loss, mean_iou_score, mean_dice_score
    
    def val_one_epochs(self, epoch):
        self.model.eval()
        val_loss, iou_score, dice_score = 0, [], []
        with Progress(SpinnerColumn(), TextColumn("[pogress.description]{task.description}"), BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as progress:
            task = progress.add_task("[red] Validating...", total=len(self.val_loader))

            for batch in self.val_loader:
                img = batch['image'].float().to(self.device)
                mask = batch['mask'].float().to(self.device)

                mask_pred = self.model(img)
                mask_pred = torch.sigmoid(mask_pred)

                loss = self.citerion(mask_pred, mask)
                val_loss+=loss.item()

                iou_score.append(IoU()(mask_pred, mask).item())
                dice_score.append(Dice_Coeff()(mask_pred, mask).item())

                progress.advance(task)
                mean_val_loss = val_loss/len(self.val_loader)
                mean_iou_score = np.mean(iou_score) * 100
                mean_dice_score = np.mean(dice_score) * 100

        return mean_val_loss, mean_iou_score, mean_dice_score
    
    def run_pipeline(self):
        log_path = os.path.join('./checkpoints', self.config['loss_filename'])

        for ep in range(self.config['epochs']):
            self.console.rule(f"[bold blue]Epoch {ep+1}/{self.config['epochs']}[/bold blue]")

            train_loss, mean_train_iou, mean_train_dice = self.train_one_epochs(ep)
            val_loss, mean_val_iou, mean_val_dice = self.val_one_epochs(ep)

            self.visualizer.print_current_losses(ep+1, len(self.train_loader), train_loss, self.config['lr'], isVal=False)
            table = Table(title=f"Epoch {ep + 1} Summary", show_lines=True)
            table.add_column("Metric", justify="center")
            table.add_column("Value", justify="center")
            table.add_row("Train Loss", f"[green]{train_loss:.4f}[/green]")
            table.add_row("Val Loss", f"[red]{val_loss:.4f}[/red]")
            table.add_row("Learning Rate", f"{self.config['lr']:.6f}")
            self.console.print(table)

            print(f"Train: IoU = {mean_train_iou:.2f}, Dice = {mean_train_dice:.2f}%")
            print(f"Val:   IoU = {mean_val_iou:.2f}%, Dice = {mean_val_dice:.2f}%")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = ep+1

                torch.save({'model_weights': self.model.state_dict(), 'val_loss': self.best_val_loss}, self.config['save_model'])
                self.console.print(f"[bold magenta] New best model save with val_loss={self.best_val_loss:.5f}[/bold magenta]")
                with open(log_path, "a") as f:
                    f.write(f"New best model saved, val_loss={self.best_val_loss:.5f}\n")
            self.scheduler.step(val_loss)

        self.console.print(f"[bold cyan]Training Finished. Best val_loss: {self.best_val_loss}, at {self.best_epoch}[/bold cyan]")
        self.visualizer.print_end(self.best_epoch, self.best_val_loss)
        torch.save({'model_weights': self.model.state_dict(), 'val_loss': self.best_val_loss}, self.config['saved_model_final'])


if __name__ == "__main__":
    with open("configs/config_crack.yml", "r") as f:
        config = yaml.safe_load(f)
    pipeline = TrainPipeline(config=config)
    pipeline.run_pipeline()

