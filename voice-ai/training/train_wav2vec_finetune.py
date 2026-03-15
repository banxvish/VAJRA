"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 Two-Stage Fine-Tuning
================================================================
Fine-tunes the pretrained superb/wav2vec2-base-superb-sid classifier
on ASVspoof data (or any custom deepfake audio dataset).

Features:
- Two-Stage Transfer Learning
- AMP (Automatic Mixed Precision)
- Gradient Clipping
- Early Stopping
- Cosine Annealing Learning Rate Scheduler
- Model Checkpointing
"""

import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets.wav2vec_loader import get_wav2vec_dataloaders

logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_feature_extractor(model):
    """Freeze CNN layers to preserve generic acoustic features."""
    for param in model.wav2vec2.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.wav2vec2.feature_projection.parameters():
        param.requires_grad = False


def freeze_transformer(model):
    """Freeze the transformer block for Stage 1 (train only classifier)."""
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = False


def unfreeze_transformer_top_layers(model, num_layers=4):
    """Unfreeze the top N transformer layers for Stage 2."""
    encoder_layers = model.wav2vec2.encoder.layers
    total_layers = len(encoder_layers)
    start_idx = max(0, total_layers - num_layers)
    
    for i, layer in enumerate(encoder_layers):
        if i >= start_idx:
            for param in layer.parameters():
                param.requires_grad = True


class Wav2VecTrainer:
    def __init__(self, model, train_loader, val_loader, device, checkpoint_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, optimizer, scheduler, clip_norm=1.0):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            inputs = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                
            self.scaler.scale(loss).backward()
            
            # Unscale before clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler:
                scheduler.step()
                
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return total_loss / len(self.train_loader), acc, f1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            inputs = batch["input_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return total_loss / len(self.val_loader), acc, f1

    def fit(self, stage_name, epochs, lr, patience=3):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(self.train_loader))
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_dir, f"best_wav2vec_{stage_name}.pt")
        
        print(f"\n--- Starting {stage_name.upper()} ---")
        print(f"Trainable params: {count_parameters(self.model):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss, train_acc, train_f1 = self.train_epoch(optimizer, scheduler)
            val_loss, val_acc, val_f1 = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            
            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"[*] New best validation loss! Saving checkpoint to {best_model_path}")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[!] Early stopping triggered after {patience} epochs without improvement.")
                    break
        
        # Load best weights before returning
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        return self.model


def main(train_dir="test_data", val_dir="test_data"):
    # Note: Replace train_dir/val_dir with actual ASVSpoof subsets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Pretrained Model & Processor
    model_name = "superb/wav2vec2-base-superb-sid"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    
    # 2. Dataloaders
    train_loader, val_loader = get_wav2vec_dataloaders(train_dir, val_dir, processor, batch_size=8)
    
    trainer = Wav2VecTrainer(model, train_loader, val_loader, device)
    
    # ==========================
    # STAGE 1: Train Classifier
    # ==========================
    freeze_feature_extractor(model)
    freeze_transformer(model)
    # The classification head (projector + classifier block) remains unfrozen
    
    trainer.fit(stage_name="stage1", epochs=5, lr=1e-4, patience=3)
    
    # ==========================
    # STAGE 2: Fine-tune backbone
    # ==========================
    unfreeze_transformer_top_layers(model, num_layers=4)
    # Train with much smaller learning rate
    trainer.fit(stage_name="stage2", epochs=5, lr=2e-5, patience=3)
    
    # Save Final Unified PyTorch Weights
    final_output = os.path.join("checkpoints", "wav2vec_finetuned_final.pt")
    torch.save(model.state_dict(), final_output)
    print(f"\n✅ Fine-tuning complete. Final model saved to {final_output}")

    # Optional: Also save in HuggingFace safe format
    model.save_pretrained(os.path.join("export", "wav2vec_finetuned_hf"))
    processor.save_pretrained(os.path.join("export", "wav2vec_finetuned_hf"))


if __name__ == "__main__":
    main()
