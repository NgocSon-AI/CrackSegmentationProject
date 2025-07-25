
from __future__ import division
import copy
import torch.optim as optim
from utils.utils import *
from pathlib import Path
from data_loader import Crack_loader
from torch.utils.data import DataLoader
from rich.console import Console
from models.model import TransMUNet
from ptflops import get_model_complexity_info
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from utils.metrics import IoU, Dice_Coeff

setup_seed(42)
console = Console()
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = float(config['lr'])

data_tra_path = config['path_to_tradata']
data_val_path = config['path_to_valdata']

DIR_IMG_tra  = os.path.join(data_tra_path, 'images')
DIR_MASK_tra = os.path.join(data_tra_path, 'masks')

DIR_IMG_val  = os.path.join(data_val_path, 'images')
DIR_MASK_val = os.path.join(data_val_path, 'masks')

img_names_tra  = [path.name for path in Path(DIR_IMG_tra).glob('*.jpg')]
mask_names_tra = [path.name for path in Path(DIR_MASK_tra).glob('*.png')]

img_names_val  = [path.name for path in Path(DIR_IMG_val).glob('*.jpg')]
mask_names_val = [path.name for path in Path(DIR_MASK_val).glob('*.png')]

train_dataset = Crack_loader(img_dir=DIR_IMG_tra, img_fnames=img_names_tra, mask_dir=DIR_MASK_tra, mask_fnames=mask_names_tra, isTrain=True)
valid_dataset = Crack_loader(img_dir=DIR_IMG_val, img_fnames=img_names_val, mask_dir=DIR_MASK_val, mask_fnames=mask_names_val, resize=True)
print(f'train_dataset:{len(train_dataset)}')
print(f'valiant_dataset:{len(valid_dataset)}')

train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True,  drop_last=True, num_workers=2, pin_memory=True)
val_loader    = DataLoader(valid_dataset, batch_size = int(config['batch_size_va']), shuffle= False, drop_last=True, num_workers=2, pin_memory=True)

model = TransMUNet(n_classes=1)
Net = model.to(device)
best_val_loss = np.inf
best = 0
if config['pretrained']:
    checkpoint = torch.load(config['saved_model'], map_location='cpu')
    Net.load_state_dict(checkpoint['model_weights'], strict=False)
    best_val_loss = checkpoint['val_loss']

optimizer = optim.Adam(Net.parameters(), lr=float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config['patience'])
criteria = DiceBCELoss()

flops, params = get_model_complexity_info(Net, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
print('flops: ', flops, 'params: ', params)
message = 'flops:%s, params:%s' % (flops, params)
# visual
visualizer = Visualizer(isTrain=True)
log_path = os.path.join('./checkpoints', config['loss_filename'])
with open(log_path, "a") as log_file:
            log_file.write('%s\n' % message)
# -------------------- Training Loop --------------------



for ep in range(config['epochs']):
    Net.train()
    train_loss, val_loss = 0, 0
    dice_train, iou_train, dice_val, iou_val = [], [], [], []

    console.rule(f"[bold blue]Epoch {ep+1}/{config['epochs']}[/bold blue]")

    # === TRAINING ===
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as progress:
        task = progress.add_task("[green]Training...", total=len(train_loader))

        for batch in train_loader:
            img = batch['image'].float().to(device)
            msk = batch['mask'].float().to(device)
            boundary = batch['boundary'].float().to(device)

            msk_pred, bnd_pred = Net(img, isTrain=True)
            msk_pred = torch.sigmoid(msk_pred)
            bnd_pred = torch.sigmoid(bnd_pred)

            loss_main = criteria(msk_pred, msk)
            loss_bnd = criteria(bnd_pred, boundary)
            loss = 0.8 * loss_main + 0.2 * loss_bnd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            dice_train.append(Dice_Coeff()(msk_pred, msk).item())
            iou_train.append(IoU()(msk_pred, msk).item())
            progress.advance(task)

    # === VALIDATION ===
    Net.eval()
    with torch.no_grad(), Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                                   BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn()) as progress:
        task = progress.add_task("[red]Validating...", total=len(val_loader))

        for batch in val_loader:
            img = batch['image'].float().to(device)
            msk = batch['mask'].float().to(device)

            msk_pred = Net(img)
            msk_pred = torch.sigmoid(msk_pred)

            loss = criteria(msk_pred, msk)
            val_loss += loss.item()
            dice_val.append(Dice_Coeff()(msk_pred, msk).item())
            iou_val.append(IoU()(msk_pred, msk).item())
            progress.advance(task)

    # === Summary ===
    train_loss_avg = train_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)
    dice_train_avg = 100 * np.mean(dice_train)
    iou_train_avg = 100 * np.mean(iou_train)
    dice_val_avg = 100 * np.mean(dice_val)
    iou_val_avg = 100 * np.mean(iou_val)

    visualizer.print_current_losses(epoch=ep+1, iters=len(train_loader), loss=train_loss_avg, lr=lr, isVal=False)
    print(f"Train: Dice = {dice_train_avg:.2f}%, IoU = {iou_train_avg:.2f}%")
    print(f"Val:   Dice = {dice_val_avg:.2f}%, IoU = {iou_val_avg:.2f}%")

    # === Table ===
    table = Table(title=f"Epoch {ep+1} Summary", show_lines=True)
    table.add_column("Metric", justify="center")
    table.add_column("Value", justify="center")
    table.add_row("Train Loss", f"[green]{train_loss_avg:.4f}[/green]")
    table.add_row("Val Loss", f"[red]{val_loss_avg:.4f}[/red]")
    table.add_row("Learning Rate", f"{lr:.6f}")
    console.print(table)

    # === Save Best ===
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        best = ep + 1
        torch.save({'model_weights': Net.state_dict(), 'val_loss': best_val_loss}, config['saved_model'])
        console.print(f"[bold magenta]New best model saved with val_loss={best_val_loss:.6f}[/bold magenta]")
        with open(log_path, "a") as f:
            f.write(f"New best model saved, val_loss={best_val_loss:.6f}\n")

    scheduler.step(val_loss_avg)

# === Save Final ===
console.print(f"[bold cyan]Training Finished. Best val_loss: {best_val_loss:.6f} at epoch {best}[/bold cyan]")
visualizer.print_end(best, best_val_loss)
torch.save({'model_weights': Net.state_dict(), 'val_loss': best_val_loss}, config['saved_model_final'])
