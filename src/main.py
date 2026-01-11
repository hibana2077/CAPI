import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Add current directory to path so we can import modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ufgvc import UFGVCDataset
try:
    from model import GeometricAwareModel
    from capi import CAPILoss
except ImportError:
    from src.model import GeometricAwareModel
    from src.capi import CAPILoss

def get_args():
    parser = argparse.ArgumentParser(description='CAPI Training')
    parser.add_argument('--dataset', type=str, default='cub_200_2011', help='Dataset name')
    parser.add_argument('--root', type=str, default='./data', help='Data root')
    parser.add_argument('--model', type=str, default='resnet50', help='Backbone model name')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained backbone')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--download', action='store_true', default=False, help='Download dataset')
    
    # CAPI specific
    parser.add_argument('--capi_dim', type=int, default=32, help='Dimension of Lie Algebra matrix (m)')
    parser.add_argument('--lambda_lie', type=float, default=1.0, help='Weight for Lie Algebra Loss')
    parser.add_argument('--gamma', type=float, default=1.0, help='Margin for separation in Lie space')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, criterion_ce, criterion_lie, optimizer, device, epoch, args):
    model.train()
    
    total_loss = 0.0
    total_ce = 0.0
    total_lie = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits, x_lie = model(images)
        
        # Cross Entropy Loss
        loss_ce = criterion_ce(logits, labels)
        
        # Lie Algebra Loss
        loss_lie = criterion_lie(x_lie, labels)
        
        # Total Loss
        loss = loss_ce + args.lambda_lie * loss_lie
        
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_lie += loss_lie.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # No tqdm, but maybe periodic print?
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, Lie: {loss_lie.item():.4f}) "
                  f"Acc: {100.*correct/total:.2f}%")
            
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    print(f"\nTrain Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%, Time: {epoch_time:.1f}s")
    return avg_loss, avg_acc

def validate(model, loader, criterion_ce, criterion_lie, device, epoch, args):
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            logits, x_lie = model(images)
            
            loss_ce = criterion_ce(logits, labels)
            # Optional: calculate Lie loss in val just for tracking
            # loss_lie = criterion_lie(x_lie, labels) 
            
            loss = loss_ce # Validation metric usually just accuracy or CE
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    avg_loss = total_loss / len(loader)
    avg_acc = 100. * correct / total
    
    print(f"Test Epoch {epoch}: Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    return avg_loss, avg_acc

def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    # Standard ImageNet transforms usually used with ResNet
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), # Resize a bit larger
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Datasets
    print(f"Loading dataset: {args.dataset}")
    try:
        train_dataset = UFGVCDataset(
            dataset_name=args.dataset,
            root=args.root,
            split='train',
            transform=transform_train,
            download=args.download
        )
        
        # Some datasets split might be 'val' or 'test'. 
        # CUB usually has train/test. We interpret 'test' as validation for determining 'best'.
        # Try finding available splits
        splits = UFGVCDataset.get_dataset_splits(args.dataset, args.root)
        test_split = 'test' if 'test' in splits else 'val'
        if test_split not in splits:
             # Fallback if split detection fails (maybe file not there yet, handled by UFGVC generic logic)
             test_split = 'test'

        test_dataset = UFGVCDataset(
            dataset_name=args.dataset,
            root=args.root,
            split=test_split,
            transform=transform_test,
            download=args.download
        )
        
        print(f"Train set: {len(train_dataset)} samples")
        print(f"Test set ({test_split}): {len(test_dataset)} samples")
        
        num_classes = len(train_dataset.classes)
        print(f"Num classes: {num_classes}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True # drop_last for LieLoss batch stability
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    print(f"Creating model {args.model} with CAPI...")
    model = GeometricAwareModel(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        capi_dim=args.capi_dim
    ).to(device)
    
    # Optimizers
    # Separate learning info for backbone and heads? Often useful but simple SGD is requested.
    # CAPI head might need distinct LR? for now global.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_lie = CAPILoss(lambda_lie=args.lambda_lie, gamma=args.gamma)
    
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_lie, optimizer, device, epoch, args
        )
        
        test_loss, test_acc = validate(
            model, test_loader, criterion_ce, criterion_lie, device, epoch, args
        )
        
        scheduler.step()
        
        if test_acc > best_acc:
            print(f"new best test acc: {test_acc:.2f}% (was {best_acc:.2f}%)")
            best_acc = test_acc
            
            # Optional: save model
            save_path = f"best_capi_{args.dataset}.pth"
            torch.save(model.state_dict(), save_path)
    
    print(f"Training complete. Best Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
