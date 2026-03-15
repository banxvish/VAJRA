"""
KAVACHA AI Voice Defense Engine — Wav2Vec2 Evaluation Pipeline
==============================================================
Evaluates fine-tuned Wav2Vec2 models on validation/test datasets.
Calculates deepfake-specific metrics:
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC
- Equal Error Rate (EER)
"""

import os
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets.wav2vec_loader import ASVSpoofWav2VecDataset
from torch.utils.data import DataLoader

def calculate_eer(y_true, y_scores):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # EER is the point where False Positive Rate equals False Negative Rate
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    return eer

def evaluate_model(model_path, test_dir="test_data", device="cuda"):
    print(f"Loading Fine-Tuned Model from {model_path}...")
    
    # HuggingFace PreTrained Format
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    # Load Test Data
    print(f"Loading test datasets from {test_dir}...")
    test_ds = ASVSpoofWav2VecDataset(test_dir, processor)
    if len(test_ds) == 0:
        print("No test data found. Exiting evaluation.")
        return
        
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    all_labels = []
    all_preds = []
    all_probs = [] # Probabilities for Class 1 (FAKE) needed for AUC and EER

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(inputs)
            logits = outputs.logits
            
            # Get Probabilities via Softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            fake_probs = probs[:, 1] # Probability of being FAKE
            
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(fake_probs.cpu().numpy())

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Need probabilities for AUC / EER
    try:
        auc = roc_auc_score(all_labels, all_probs)
        eer = calculate_eer(all_labels, all_probs)
    except Exception as e:
        print(f"Skipping AUC/EER calculation: {str(e)}")
        auc, eer = 0.0, 0.0

    print("\n" + "="*40)
    print(f"WAV2VEC2 EVALUATION RESULTS")
    print("="*40)
    print(f"Dataset Size: {len(test_ds)} audio clips")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"EER:       {eer * 100:.2f}%")
    print("="*40)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Point this to the finetuned HuggingFace folder exported during training
    evaluate_model(model_path="export/wav2vec_finetuned_hf", test_dir="test_data", device=device)
    
