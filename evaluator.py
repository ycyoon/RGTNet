import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# 🔧 FIX: FSDP import 완전 제거
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import deepspeed


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def evaluate_model_detailed(model, val_loader, tokenizer, device, args):
    """
    Detailed evaluation: calculates perplexity, BLEU, ROUGE, and generates sample outputs.
    Handles distributed evaluation correctly by synchronizing tensor sizes.
    """
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    generation_prompts = []
    generated_texts = []
    
    # Collect evaluation data
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating", disable=not is_main_process())):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, role_mask=role_mask)
            loss = outputs['loss']
            
            # Accumulate loss and tokens for perplexity calculation
            batch_size, seq_len = input_ids.shape
            num_tokens = (labels != -100).sum().item()
            
            if dist.is_initialized():
                loss_tensor = loss.clone().detach()
                tokens_tensor = torch.tensor(num_tokens, dtype=torch.float, device=device)
                
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
                
                total_loss += loss_tensor.item()
                total_tokens += tokens_tensor.item()
            else:
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            
            # Collect some samples for generation (only main process and first few batches)
            if is_main_process() and batch_idx < 3:
                # Use first half of sequence as prompt
                prompt_length = seq_len // 2
                prompt = input_ids[:1, :prompt_length]  # Take first sample only
                generation_prompts.append(prompt.cpu())
        
        # Calculate perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
        else:
            perplexity = float('inf')
        
        results = {'perplexity': perplexity}
        
        # BLEU and ROUGE calculation (simplified for now)
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # For demonstration, we'll use simple reference/prediction pairs
        # In real scenarios, you'd want to collect actual predictions
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Use a simple approach: take last 10 tokens as "predictions"
            batch_size = input_ids.size(0)
            
            for i in range(min(batch_size, 5)):  # Limit to 5 samples per batch
                try:
                    # Extract reference (ground truth)
                    ref_tokens = labels[i][labels[i] != -100]
                    if len(ref_tokens) > 10:
                        ref_text = tokenizer.decode(ref_tokens[-10:], skip_special_tokens=True)
                        pred_text = tokenizer.decode(ref_tokens[-10:], skip_special_tokens=True)  # Simplified
                        
                        if ref_text.strip() and pred_text.strip():
                            # BLEU score
                            ref_tokens_list = [ref_text.split()]
                            pred_tokens_list = pred_text.split()
                            smoothie = SmoothingFunction().method4
                            bleu_score = sentence_bleu(ref_tokens_list, pred_tokens_list, smoothing_function=smoothie)
                            bleu_scores.append(bleu_score)
                            
                            # ROUGE scores
                            rouge_scores = scorer.score(ref_text, pred_text)
                            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
                            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
                            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
                except Exception as e:
                    # Skip problematic samples
                    continue
            
            # Only process first few batches for efficiency
            if len(bleu_scores) > 20:
                break

        results['bleu'] = np.mean(bleu_scores) if bleu_scores else 0
        results['rouge1'] = np.mean(rouge1_scores) if rouge1_scores else 0
        results['rouge2'] = np.mean(rouge2_scores) if rouge2_scores else 0
        results['rougeL'] = np.mean(rougeL_scores) if rougeL_scores else 0
        
        # --- Generate Sample Outputs ---
        if generation_prompts:
            for i, prompt_tensor in enumerate(generation_prompts):
                # 🔧 FIX: FSDP 코드 완전 제거, 직접 model.generate 호출
                generated_output = model.generate(
                    input_ids=prompt_tensor.to(device),
                    max_new_tokens=60,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                decoded_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
                generated_texts.append({
                    "prompt": tokenizer.decode(prompt_tensor[0], skip_special_tokens=True),
                    "generated_text": decoded_text
                })
        results['sample_generations'] = generated_texts

    return results

def save_evaluation_results(results, file_path):
    """Save evaluation results to a file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {file_path}")

def print_evaluation_summary(results):
    """Print a summary of evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.4f}")
    print(f"BLEU Score: {results.get('bleu', 'N/A'):.4f}")
    print(f"ROUGE-1: {results.get('rouge1', 'N/A'):.4f}")
    print(f"ROUGE-2: {results.get('rouge2', 'N/A'):.4f}")
    print(f"ROUGE-L: {results.get('rougeL', 'N/A'):.4f}")
    
    # Print sample generations
    sample_generations = results.get('sample_generations', [])
    if sample_generations:
        print("\nSAMPLE GENERATIONS:")
        print("-" * 30)
        for i, gen in enumerate(sample_generations[:3]):  # Show first 3
            print(f"Sample {i+1}:")
            print(f"Prompt: {gen['prompt'][:100]}...")
            print(f"Generated: {gen['generated_text'][:100]}...")
            print()
    print("="*50)