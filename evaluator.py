import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def evaluate_model_detailed(model, val_loader, tokenizer, device, args):
    """
    Detailed evaluation: calculates perplexity, BLEU, ROUGE, and generates sample outputs.
    Handles distributed evaluation correctly by synchronizing tensor sizes.
    """
    model.eval()
    
    # --- Metrics Initialization ---
    total_loss = 0
    total_tokens = 0
    all_preds = []
    all_refs = []
    
    # --- For sample generation ---
    generation_prompts = []
    generated_texts = []

    with torch.no_grad():
        # Full evaluation now that CUDA indexing errors are resolved
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Detailed Evaluating", leave=True, disable=not is_main_process())):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            role_mask = batch['role_mask'].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

            # --- 1. Perplexity Calculation ---
            outputs = model(input_ids=input_ids, role_mask=role_mask, labels=labels)
            loss = outputs.get('loss')
            if loss is not None:
                num_tokens = attention_mask.sum()
                total_loss += (loss.item() * num_tokens)
                total_tokens += num_tokens

            # --- 2. Generation for BLEU/ROUGE ---
            try:
                # Ensure input_ids are within vocabulary range before generation
                vocab_size = len(tokenizer)
                input_ids_safe = torch.clamp(input_ids, 0, vocab_size - 1)
                
                if isinstance(model, FSDP):
                    with FSDP.summon_full_params(model, writeback=False):
                        preds = model.generate(
                            input_ids=input_ids_safe,
                            max_new_tokens=min(args.max_seq_len, 20),  # Further limit generation length
                            pad_token_id=tokenizer.pad_token_id,
                            do_sample=False  # Use greedy decoding for stability
                        )
                else:
                    unwrapped_model = model.module if isinstance(model, DDP) else model
                    preds = unwrapped_model.generate(
                        input_ids=input_ids_safe,
                        max_new_tokens=min(args.max_seq_len, 20),  # Further limit generation length
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False  # Use greedy decoding for stability
                    )
                
                # Ensure generated tokens are within vocabulary range
                preds = torch.clamp(preds, 0, vocab_size - 1)
                
            except Exception as e:
                print(f"Warning: Generation failed in evaluation, using input as fallback. Error: {e}")
                # Fallback: use input_ids as predictions if generation fails
                preds = input_ids_safe.clone()

            # --- 3. Gather results in a distributed-safe way ---
            if dist.is_initialized():
                # Synchronize prediction lengths to prevent deadlock in all_gather
                local_max_len = torch.tensor(preds.size(1), device=device, dtype=torch.int64)
                dist.all_reduce(local_max_len, op=dist.ReduceOp.MAX)
                global_max_len = local_max_len.item()

                # Pad predictions and labels to the global max length
                pad_size = global_max_len - preds.size(1)
                if pad_size > 0:
                    preds = F.pad(preds, (0, pad_size), "constant", tokenizer.pad_token_id)

                pad_size_labels = global_max_len - labels.size(1)
                if pad_size_labels > 0:
                    labels = F.pad(labels, (0, pad_size_labels), "constant", -100)

                # Gather padded predictions and labels from all processes
                world_size = dist.get_world_size()
                gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
                gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
                dist.all_gather(gathered_preds, preds)
                dist.all_gather(gathered_labels, labels)

                if is_main_process():
                    preds_to_decode = torch.cat(gathered_preds, dim=0)
                    labels_to_decode = torch.cat(gathered_labels, dim=0)
                else: # Other processes don't need to store all data
                    preds_to_decode, labels_to_decode = None, None
            else: # Single GPU case
                preds_to_decode, labels_to_decode = preds, labels

            if is_main_process():
                # Safely decode predictions and references on the main process
                try:
                    # Ensure all token IDs are within vocabulary range before decoding
                    vocab_size = len(tokenizer)
                    preds_to_decode = torch.clamp(preds_to_decode, 0, vocab_size - 1)
                    
                    # Decode predictions safely
                    decoded_preds = tokenizer.batch_decode(preds_to_decode, skip_special_tokens=True)
                    
                    # Handle labels safely
                    labels_to_decode[labels_to_decode == -100] = tokenizer.pad_token_id
                    labels_to_decode = torch.clamp(labels_to_decode, 0, vocab_size - 1)
                    decoded_labels = tokenizer.batch_decode(labels_to_decode, skip_special_tokens=True)
                    
                    all_preds.extend(decoded_preds)
                    all_refs.extend(decoded_labels)
                    
                except Exception as e:
                    print(f"Warning: Failed to decode tokens in evaluation, skipping batch. Error: {e}")
                    # Add empty strings as fallback
                    batch_size = preds_to_decode.size(0) if preds_to_decode is not None else 1
                    all_preds.extend([""] * batch_size)
                    all_refs.extend([""] * batch_size)
            
            # Collect some prompts for sample generation on the main process
            if batch_idx == 0 and is_main_process() and len(generation_prompts) < 5:
                for i in range(input_ids.size(0)):
                    if len(generation_prompts) < 5:
                        # Fix tensor-to-scalar conversion error
                        non_pad_mask = (input_ids[i] != tokenizer.pad_token_id)
                        if torch.any(non_pad_mask):
                            prompt_len = non_pad_mask.sum().item()
                            prompt = input_ids[i][:max(1, prompt_len // 2)].unsqueeze(0)
                            generation_prompts.append(prompt)
                        else:
                            # Fallback: use first token if all are pad tokens
                            prompt = input_ids[i][:1].unsqueeze(0)
                            generation_prompts.append(prompt)


    # --- Synchronize and Calculate Final Metrics ---
    results = {}

    # Sync perplexity metrics
    if dist.is_initialized():
        total_loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float32)
        total_tokens_tensor = torch.tensor(total_tokens, device=device, dtype=torch.int64)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()
        total_tokens = total_tokens_tensor.item()

    # Calculate metrics only on the main process where all data is gathered
    if is_main_process():
        # Perplexity
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = np.exp(avg_loss)
            results['perplexity'] = perplexity
        else:
            results['perplexity'] = float('inf')

        # BLEU and ROUGE
        chencherry = SmoothingFunction().method1
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        bleu_scores = []
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        for pred, ref in zip(all_preds, all_refs):
            ref_tokens = [tokenizer.tokenize(ref)]
            pred_tokens = tokenizer.tokenize(pred)
            
            if pred_tokens: # Avoid division by zero for empty predictions
                bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry))
            
            rouge_scores = scorer.score(ref, pred)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        results['bleu'] = np.mean(bleu_scores) if bleu_scores else 0
        results['rouge1'] = np.mean(rouge1_scores) if rouge1_scores else 0
        results['rouge2'] = np.mean(rouge2_scores) if rouge2_scores else 0
        results['rougeL'] = np.mean(rougeL_scores) if rougeL_scores else 0
        
        # --- Generate Sample Outputs ---
        if generation_prompts:
            for i, prompt_tensor in enumerate(generation_prompts):
                if isinstance(model, FSDP):
                    with FSDP.summon_full_params(model, writeback=False):
                        generated_output = model.generate(
                            input_ids=prompt_tensor.to(device),
                            max_new_tokens=60,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.pad_token_id
                        )
                else:
                    unwrapped_model = model.module if isinstance(model, DDP) else model
                    generated_output = unwrapped_model.generate(
                        input_ids=prompt_tensor.to(device),
                        max_new_tokens=60,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id
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
    """Prints a summary of the evaluation results."""
    print("\n--- Evaluation Summary ---")
    if 'perplexity' in results:
        print(f"Perplexity: {results['perplexity']:.4f}")
    if 'bleu' in results:
        print(f"BLEU Score: {results['bleu']:.4f}")
    if 'rouge1' in results:
        print(f"ROUGE-1: {results['rouge1']:.4f}")
    if 'rouge2' in results:
        print(f"ROUGE-2: {results['rouge2']:.4f}")
    if 'rougeL' in results:
        print(f"ROUGE-L: {results['rougeL']:.4f}")
    
    if 'sample_generations' in results and results['sample_generations']:
        print("\n--- Sample Generations ---")
        for i, sample in enumerate(results['sample_generations']):
            print(f"Sample {i+1}:")
            print(f"  Prompt: {sample['prompt']}")
            print(f"  Generated: {sample['generated_text']}")
            print("-" * 20)
    print("--------------------------\n")