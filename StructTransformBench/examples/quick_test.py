#!/usr/bin/env python3
"""
Quick test script to debug the multi-model benchmark issues
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

def test_model_generation():
    print("=== Quick Model Generation Test ===")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model.eval()
        print("✅ Model loaded")
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Tell me a short joke."
        ]
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            
            # Format prompt
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                print(f"Formatted: {formatted_prompt[:100]}...")
            else:
                formatted_prompt = f"User: {prompt}\nAssistant: "
                print(f"Simple format: {formatted_prompt}")
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            print(f"Response: '{response}'")
            print(f"Length: {len(response)}")
            
            if not response:
                print("❌ Empty response!")
            else:
                print("✅ Valid response")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_model_generation()
