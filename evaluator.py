import torch
import json
from tqdm import tqdm

def evaluate_model_detailed(model, head, eval_loader, device, args):
    """Detailed evaluation of the model"""
    model.eval()
    head.eval()
    
    results = {
        'total_samples': 0,
        'total_loss': 0,
        'predictions': [],
        'ground_truth': [],
        'sample_outputs': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            role_mask = batch['role_mask'].to(device)
            labels = batch['labels'].to(device)
            original_texts = batch['original_texts']
            
            # Forward pass
            src_key_padding_mask = (attention_mask == 0)
            outputs = model(input_ids, role_mask, src_key_padding_mask=src_key_padding_mask)
            pooled_output = outputs.mean(dim=1)
            logits = head(pooled_output)
            
            # Calculate loss
            loss = torch.nn.MSELoss()(logits.squeeze(), labels.float())
            results['total_loss'] += loss.item()
            results['total_samples'] += len(labels)
            
            # Store predictions and ground truth
            predictions = logits.squeeze().cpu().numpy()
            ground_truth = labels.cpu().numpy()
            
            # Handle scalar predictions (when batch size is 1)
            if predictions.ndim == 0:
                predictions = [float(predictions)]
            else:
                predictions = predictions.tolist()
            
            if ground_truth.ndim == 0:
                ground_truth = [float(ground_truth)]
            else:
                ground_truth = ground_truth.tolist()
            
            results['predictions'].extend(predictions)
            results['ground_truth'].extend(ground_truth)
            
            # Store sample outputs for analysis
            if batch_idx < 5:  # Store first 5 batches for analysis
                for i in range(min(3, len(original_texts))):  # First 3 samples per batch
                    pred_val = predictions[i] if i < len(predictions) else predictions[0]
                    results['sample_outputs'].append({
                        'input_text': original_texts[i][:200] + "..." if len(original_texts[i]) > 200 else original_texts[i],
                        'prediction': float(pred_val),
                        'ground_truth': float(ground_truth[i])
                    })
    
    # Calculate final metrics
    results['average_loss'] = results['total_loss'] / len(eval_loader)
    
    # Add basic statistics
    if results['predictions']:
        import numpy as np
        pred_array = np.array(results['predictions'])
        gt_array = np.array(results['ground_truth'])
        
        results['prediction_stats'] = {
            'mean': float(np.mean(pred_array)),
            'std': float(np.std(pred_array)),
            'min': float(np.min(pred_array)),
            'max': float(np.max(pred_array))
        }
        
        results['ground_truth_stats'] = {
            'mean': float(np.mean(gt_array)),
            'std': float(np.std(gt_array)),
            'min': float(np.min(gt_array)),
            'max': float(np.max(gt_array))
        }
        
        # Calculate correlation if possible
        if np.std(pred_array) > 0 and np.std(gt_array) > 0:
            results['correlation'] = float(np.corrcoef(pred_array, gt_array)[0, 1])
    
    model.train()
    head.train()
    
    return results

def save_evaluation_results(results, filepath):
    """Save evaluation results to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {filepath}")

def print_evaluation_summary(results):
    """Print a summary of evaluation results"""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total samples: {results['total_samples']}")
    print(f"Average loss: {results['average_loss']:.4f}")
    
    if 'prediction_stats' in results:
        print(f"Prediction stats: mean={results['prediction_stats']['mean']:.4f}, std={results['prediction_stats']['std']:.4f}")
        print(f"Ground truth stats: mean={results['ground_truth_stats']['mean']:.4f}, std={results['ground_truth_stats']['std']:.4f}")
        
        if 'correlation' in results:
            print(f"Correlation: {results['correlation']:.4f}")
    
    if results['sample_outputs']:
        print("\nSample outputs:")
        for i, sample in enumerate(results['sample_outputs'][:3]):
            print(f"\nSample {i+1}:")
            print(f"Input: {sample['input_text']}")
            print(f"Prediction: {sample['prediction']:.4f}")
            print(f"Ground truth: {sample['ground_truth']:.4f}")
    
    print("="*50)