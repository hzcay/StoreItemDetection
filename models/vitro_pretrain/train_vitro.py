"""
Training script for ResMobileNetV2 on in-vitro dataset (pretrain phase)
Chiến lược: Train nhẹ để học đặc trưng cơ bản, có thể freeze một số layer đầu
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backbone.ResMobileNetV2 import ResMobileNetV2, res_mobilenet_conf


class VitroDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue
            
            web_dir = class_dir / "web"
            if web_dir.exists():
                jpeg_dir = web_dir / "JPEG"
                if jpeg_dir.exists():
                    for img_file in sorted(jpeg_dir.glob("*.jpg")):
                        if img_file.name.lower() != "thumbs.db":
                            self.samples.append((str(img_file), class_id))
                
                png_dir = web_dir / "PNG"
                if png_dir.exists():
                    for img_file in sorted(png_dir.glob("*.png")):
                        if img_file.name.lower() != "thumbs.db":
                            self.samples.append((str(img_file), class_id))
        
        print(f"   Loaded {len(self.samples)} samples from {len(set([s[1] for s in self.samples]))} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = label - 1

        return image, label


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
):
    """
    Tạo train/val/test loaders
    
    Args:
        train_split: Tỷ lệ train (mặc định: 0.7)
        val_split: Tỷ lệ validation (mặc định: 0.15)
        test_split: Tỷ lệ test (mặc định: 0.15)
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = VitroDataset(data_dir, transform=transform_train)
    
    # Split: train/val/test
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update transforms
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_test
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, len(set([s[1] for s in full_dataset.samples]))


def freeze_layers(model: nn.Module, freeze_stem: bool = True, freeze_mobile: bool = False):
    """
    Freeze một số layer để train nhẹ hơn
    
    Args:
        freeze_stem: Freeze ResNet stem (conv1, bn1, maxpool, transition)
        freeze_mobile: Freeze MobileNet mid-blocks
    """
    if freeze_stem:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.transition.parameters():
            param.requires_grad = False
        print("✅ Frozen: ResNet stem")
    
    if freeze_mobile:
        for param in model.mobile_features.parameters():
            param.requires_grad = False
        print("✅ Frozen: MobileNet mid-blocks")
    
    # Luôn train ResNet tail và embedding head
    print("✅ Training: ResNet tail + Embedding head")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train 1 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward: lấy embedding và logits từ ArcFace
        embeddings = model(images)
        logits = model.fc_arcface(embeddings)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="[Val]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        logits = model.fc_arcface(embeddings)
        
        loss = criterion(logits, labels)
        
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    """Extract embeddings và labels từ data_loader"""
    model.eval()
    embeddings = []
    labels = []
    
    for images, batch_labels in tqdm(data_loader, desc="Extracting embeddings"):
        images = images.to(device)
        batch_embeddings = model(images)
        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels)
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return embeddings, labels


def compute_recall_at_k(test_emb, gallery_emb, test_labels, gallery_labels, k=5, device="cpu"):
    """Tính Recall@k"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Get top-k
    topk = torch.topk(similarity, k=k, dim=1).indices
    
    # Check if true label in top-k
    correct = 0
    for i in range(len(test_labels)):
        true_label = test_labels[i]
        retrieved_labels = gallery_labels[topk[i]]
        if (retrieved_labels == true_label).any():
            correct += 1
    
    recall = correct / len(test_labels)
    return recall


def compute_precision_at_1(test_emb, gallery_emb, test_labels, gallery_labels, device="cpu"):
    """Tính Precision@1 (1-NN Accuracy)"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Get top-1
    _, top1 = torch.topk(similarity, k=1, dim=1)
    top1 = top1.squeeze(1)
    
    # Check if predicted label matches true label
    predicted_labels = gallery_labels[top1]
    correct = (predicted_labels == test_labels).sum().item()
    precision = correct / len(test_labels)
    
    return precision


def compute_mean_average_precision(test_emb, gallery_emb, test_labels, gallery_labels, device="cpu"):
    """Tính Mean Average Precision (mAP)"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Sort by similarity
    _, indices = torch.sort(similarity, dim=1, descending=True)
    
    # Compute AP for each query
    aps = []
    for i in range(len(test_labels)):
        query_label = test_labels[i]
        retrieved_labels = gallery_labels[indices[i]]
        
        # Find relevant items (same label)
        relevant = (retrieved_labels == query_label).float()
        
        if relevant.sum() == 0:
            aps.append(0.0)
            continue
        
        # Compute precision at each relevant position
        cumulative_relevant = torch.cumsum(relevant, dim=0)
        precision_at_k = cumulative_relevant / torch.arange(1, len(relevant) + 1, dtype=torch.float32, device=device)
        
        # Average precision
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap.item())
    
    map_score = sum(aps) / len(aps)
    return map_score


@torch.no_grad()
def evaluate_test(model, test_loader, train_loader, criterion, device):
    """Evaluate on test set với các metrics: Precision@1, Recall@5, mAP"""
    model.eval()
    
    print("\n🧪 Evaluating on TEST set...")
    print("   Extracting embeddings...")
    
    # Extract embeddings từ test set
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    
    # Extract embeddings từ train set (dùng làm gallery)
    gallery_embeddings, gallery_labels = extract_embeddings(model, train_loader, device)
    
    print("   Computing metrics...")
    
    # Compute metrics
    precision_at_1 = compute_precision_at_1(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, device=device
    )
    
    recall_at_5 = compute_recall_at_k(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, k=5, device=device
    )
    
    map_score = compute_mean_average_precision(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, device=device
    )
    
    # Print results
    print(f"\n📊 TEST Results:")
    print(f"   1-NN Accuracy (Precision@1): {precision_at_1 * 100:.2f}%")
    print(f"   Recall@5: {recall_at_5 * 100:.2f}%")
    print(f"   Mean Average Precision (mAP): {map_score:.4f}")
    
    return {
        'precision_at_1': precision_at_1,
        'recall_at_5': recall_at_5,
        'mean_average_precision': map_score,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels,
        'gallery_embeddings': gallery_embeddings,
        'gallery_labels': gallery_labels
    }


def main():
    parser = argparse.ArgumentParser(description='Train ResMobileNetV2 on in-vitro dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to in-vitro data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--freeze-stem', action='store_true',
                        help='Freeze ResNet stem layers')
    parser.add_argument('--freeze-mobile', action='store_true',
                        help='Freeze MobileNet mid-blocks')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📦 Loading dataset...")
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"   Classes: {num_classes}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    print("\n🏗️  Creating model...")
    inverted_residual_setting, last_channel = res_mobilenet_conf(width_mult=1.0)
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=args.embedding_size,
        num_classes=num_classes
    ).to(args.device)
    
    if args.freeze_stem or args.freeze_mobile:
        freeze_layers(model, freeze_stem=args.freeze_stem, freeze_mobile=args.freeze_mobile)
    
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f"📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Device: {args.device}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}\n")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}\n")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'embedding_size': args.embedding_size,
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)\n")
    
    # Evaluate on test set sau khi training xong
    print(f"\n{'='*70}")
    print("🧪 FINAL TEST EVALUATION")
    print(f"{'='*70}")
    
    # Load best model để test
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'), map_location=args.device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_results = evaluate_test(
        model, test_loader, train_loader, criterion, args.device
    )
    
    # Lưu test results vào checkpoint
    best_checkpoint['test_precision_at_1'] = test_results['precision_at_1']
    best_checkpoint['test_recall_at_5'] = test_results['recall_at_5']
    best_checkpoint['test_mean_average_precision'] = test_results['mean_average_precision']
    torch.save(best_checkpoint, os.path.join(args.output_dir, 'best.pth'))
    
    print(f"\n🎉 Training completed!")
    print(f"   Best Val Acc: {best_val_acc:.2f}%")
    print(f"   Test Precision@1: {test_results['precision_at_1'] * 100:.2f}%")
    print(f"   Test Recall@5: {test_results['recall_at_5'] * 100:.2f}%")
    print(f"   Test mAP: {test_results['mean_average_precision']:.4f}")
    print(f"   Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

