import os
import pickle
import logging
import yaml
import json
import time
from datetime import datetime

from easyjailbreak.attacker.SchemaAttackUpdated import StructuredAttack
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

def create_model(model_config):
    """Create model from config, skip if dummy"""
    if "dummy" in model_config["model_name"].lower():
        return None
    return OpenaiModel(
        model_name=model_config["model_name"],
        base_url=model_config["base_url"],
        api_keys=os.getenv(model_config["api_key_env"], "EMPTY"),
        generation_config=model_config["generation_config"]
    )

def main():
    print("🚀 Starting Llama-3.2-3B Benchmark with Pre-generated Attacks")
    print("="*80)
    
    # Load configuration
    with open("config_llama32.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"Target Model: {config['models']['target']['model_name']}")
    print(f"Using pre-generated attack datasets (no attack model needed)")
    
    # Initialize models (skip dummy attack model)
    target_model = create_model(config["models"]["target"])
    attack_model = create_model(config["models"]["attack"])  # Will be None (dummy)
    eval_model = create_model(config["models"]["eval"])
    refusal_eval_model = create_model(config["models"]["refusal_eval"])
    
    # Load dataset and structures
    dataset = JailbreakDataset('HarmBench')
    structures_dict = config["structures"]
    save_path = config["paths"]["save_path"]
    
    # Get settings
    attack_config = config.get('attack_settings', {})
    combination_mode = attack_config.get('combination_mode', 1)
    max_queries = attack_config.get('max_queries', None)
    test_mode = attack_config.get('test_mode', False)
    
    print(f"Dataset Size: {len(dataset)}")
    print(f"Max Queries per Structure: {max_queries}")
    print(f"Test Mode: {test_mode}")
    print(f"Structures: {list(structures_dict.keys())}")
    
    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Results tracking
    results_summary = {}
    failed_indices = range(min(len(dataset), max_queries if max_queries else len(dataset)))
    
    # Test each structure
    for structure_type in structures_dict.keys():
        combinations_to_test = [False, True] if not test_mode else [False]
        
        for combined_flag in combinations_to_test:
            # Clear logging handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            # Create structure name and path
            structure_name = f"{structure_type}{'_Combined' if combined_flag else ''}"
            structure_save_path = os.path.join(save_path, structure_name)
            os.makedirs(structure_save_path, exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"TESTING: {structure_name}")
            print(f"{'='*60}")
            
            # Load pre-generated attack data
            pkl_file_path = structures_dict[structure_type]
            
            try:
                print(f"Loading pre-generated attacks from: {pkl_file_path}")
                with open(pkl_file_path, 'rb') as file:
                    temp_dict = pickle.load(file)
                print(f"✅ Loaded {len(temp_dict)} pre-generated attacks")
            except Exception as e:
                print(f"❌ Error loading attack data: {e}")
                results_summary[structure_name] = {
                    'error': f"Failed to load: {e}",
                    'asr': 0,
                    'successful_attacks': 0,
                    'total_queries': 0,
                    'time_taken': 0
                }
                continue
            
            # Prepare attack dictionary (limited by max_queries)
            attack_dict = {}
            query_limit = min(max_queries or len(temp_dict), len(temp_dict))
            
            for i, (key, value) in enumerate(temp_dict.items()):
                if i >= query_limit:
                    break
                if i in failed_indices:
                    attack_dict[key] = value
            
            print(f"Running benchmark with {len(attack_dict)} pre-generated attacks...")
            
            # Initialize attacker (no attack model needed for pre-generated attacks)
            start_time = time.time()
            
            try:
                attacker = StructuredAttack(
                    attack_model=attack_model,  # None (dummy)
                    schema_type=structure_type,
                    combination_1=combined_flag if combination_mode == 1 else False,
                    combination_2=combined_flag if combination_mode == 2 else False,
                    target_model=target_model,
                    eval_model=eval_model,
                    refusal_eval_model=refusal_eval_model,
                    jailbreak_datasets=dataset,
                    max_queries=max_queries,
                    use_mutations=False,
                    mutations_num=5,
                    use_repeated=False,
                    repeated_num=5,
                    use_same_repeated=False,
                    same_repeated_num=10,
                    parallelize=True,
                    num_threads=10,  # Reduced for stability
                    save_path=structure_save_path
                )
                
                pkl_save_path = f"{structure_save_path}/Responses.pkl"
                
                print(f"⚡ Running benchmark for {structure_name}...")
                results = attacker.attack(attack_dict, pkl_save_path)
                end_time = time.time()
                
                # Calculate metrics
                total_queries = len(attack_dict)
                successful_attacks = 0
                
                # Try to get success count from results
                if os.path.exists(pkl_save_path):
                    try:
                        with open(pkl_save_path, 'rb') as f:
                            response_data = pickle.load(f)
                        
                        # Count successful attacks
                        if isinstance(response_data, dict):
                            for response in response_data.values():
                                if hasattr(response, 'is_harmful') and response.is_harmful:
                                    successful_attacks += 1
                                elif isinstance(response, dict) and response.get('is_harmful', False):
                                    successful_attacks += 1
                        elif isinstance(response_data, list):
                            for response in response_data:
                                if hasattr(response, 'is_harmful') and response.is_harmful:
                                    successful_attacks += 1
                                elif isinstance(response, dict) and response.get('is_harmful', False):
                                    successful_attacks += 1
                        
                        print(f"✅ Processed {total_queries} attacks successfully")
                    except Exception as e:
                        print(f"⚠️  Could not parse results file: {e}")
                        successful_attacks = 0
                
                asr = successful_attacks / total_queries if total_queries > 0 else 0
                time_taken = end_time - start_time
                
                results_summary[structure_name] = {
                    'asr': asr,
                    'successful_attacks': successful_attacks,
                    'total_queries': total_queries,
                    'time_taken': time_taken,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"📊 Results for {structure_name}:")
                print(f"   ASR: {asr:.2%} ({successful_attacks}/{total_queries})")
                print(f"   Time: {time_taken:.1f}s")
                print(f"   Results saved to: {pkl_save_path}")
                
            except Exception as e:
                print(f"❌ Error during benchmark: {e}")
                results_summary[structure_name] = {
                    'error': str(e),
                    'asr': 0,
                    'successful_attacks': 0,
                    'total_queries': len(attack_dict),
                    'time_taken': time.time() - start_time
                }
            
            finally:
                if 'attacker' in locals():
                    try:
                        attacker.cleanup()
                    except:
                        pass
            
            print(f"✅ Completed {structure_name}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(save_path, f"llama32_benchmark_results_{timestamp}.json")
    
    benchmark_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model': config['models']['target']['model_name'],
            'method': 'pre-generated attacks',
            'max_queries_per_structure': max_queries
        },
        'results': results_summary
    }
    
    with open(summary_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("LLAMA-3.2-3B BENCHMARK FINAL SUMMARY")
    print(f"{'='*80}")
    
    total_successful = 0
    total_queries = 0
    
    print(f"{'Structure':<25} {'ASR':<8} {'Success/Total':<15} {'Time (s)':<10} {'Status'}")
    print("-" * 80)
    
    for structure_name, metrics in results_summary.items():
        if 'error' in metrics:
            print(f"{structure_name:<25} {'ERROR':<8} {'N/A':<15} {metrics['time_taken']:<10.1f} ❌")
        else:
            asr = metrics['asr']
            success = metrics['successful_attacks']
            total = metrics['total_queries']
            time_taken = metrics['time_taken']
            
            total_successful += success
            total_queries += total
            
            status = "✅" if asr > 0.1 else "⚠️" if asr > 0 else "❌"
            print(f"{structure_name:<25} {asr:<8.2%} {success:>3d}/{total:<3d} {time_taken:<10.1f} {status}")
    
    print("-" * 80)
    overall_asr = total_successful / total_queries if total_queries > 0 else 0
    print(f"{'OVERALL':<25} {overall_asr:<8.2%} {total_successful:>3d}/{total_queries:<3d}")
    print("="*80)
    
    print(f"\n🎯 Summary:")
    print(f"   • Model: meta-llama/Llama-3.2-3B-Instruct")
    print(f"   • Method: Pre-generated attack datasets")
    print(f"   • Overall ASR: {overall_asr:.2%}")
    print(f"   • Total Attacks: {total_successful}/{total_queries}")
    print(f"   • Results saved: {summary_file}")

if __name__ == "__main__":
    main()
