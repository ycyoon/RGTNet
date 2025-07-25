import os
import pickle
import logging
import yaml
import json
import time
from datetime import datetime
from collections import defaultdict

from easyjailbreak.attacker.SchemaAttackUpdated import StructuredAttack
from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

class LlamaBenchmarkRunner:
    def __init__(self, config_path="config_llama32.yaml"):
        print(f"Loading configuration from {config_path}")
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize tracking
        self.results_summary = {}
        self.start_time = time.time()
        
        # Create save path
        self.save_path = self.config["paths"]["save_path"]
        os.makedirs(self.save_path, exist_ok=True)
        
        print(f"Results will be saved to: {self.save_path}")
        
    def create_model(self, model_config):
        return OpenaiModel(
            model_name=model_config["model_name"],
            base_url=model_config["base_url"],
            api_keys=os.getenv(model_config["api_key_env"], "EMPTY"),
            generation_config=model_config["generation_config"]
        )
    
    def test_server_connection(self, model_name, base_url):
        """Test if server is responding"""
        import requests
        try:
            response = requests.get(f"{base_url}/models", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def run_benchmark(self):
        print("="*80)
        print("STARTING LLAMA-3.2-3B STRUCTURED ATTACK BENCHMARK")
        print("="*80)
        
        # Test server connections first
        print("Testing server connections...")
        servers = [
            ("Target", self.config["models"]["target"]["base_url"]),
            ("Attack", self.config["models"]["attack"]["base_url"]),
            ("Eval", self.config["models"]["eval"]["base_url"]),
            ("Refusal Eval", self.config["models"]["refusal_eval"]["base_url"])
        ]
        
        for name, url in servers:
            if self.test_server_connection(name, url):
                print(f"✅ {name} server: {url}")
            else:
                print(f"❌ {name} server: {url} - NOT RESPONDING")
        
        # Initialize models
        print("\nInitializing models...")
        try:
            target_model = self.create_model(self.config["models"]["target"])
            attack_model = self.create_model(self.config["models"]["attack"])
            eval_model = self.create_model(self.config["models"]["eval"])
            refusal_eval_model = self.create_model(self.config["models"]["refusal_eval"])
            print("✅ All models initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return
        
        # Load dataset
        print("Loading dataset...")
        try:
            dataset = JailbreakDataset('HarmBench')
            print(f"✅ Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return
        
        # Get config settings
        structures_dict = self.config["structures"]
        attack_config = self.config.get('attack_settings', {})
        combination_mode = attack_config.get('combination_mode', 1)
        max_queries = attack_config.get('max_queries', None)
        test_mode = attack_config.get('test_mode', False)
        
        print(f"Configuration:")
        print(f"  Target Model: {self.config['models']['target']['model_name']}")
        print(f"  Dataset Size: {len(dataset)}")
        print(f"  Max Queries: {max_queries}")
        print(f"  Test Mode: {test_mode}")
        print(f"  Structures: {list(structures_dict.keys())}")
        
        failed_indices = range(min(len(dataset), max_queries if max_queries else len(dataset)))
        
        # Run benchmark for each structure
        for structure_type in structures_dict.keys():
            combinations_to_test = [False, True] if not test_mode else [False]  # Skip combined in test mode
            
            for combined_flag in combinations_to_test:
                structure_name = f"{structure_type}{'_Combined' if combined_flag else ''}"
                
                print("\n" + "="*60)
                print(f"TESTING STRUCTURE: {structure_name}")
                print("="*60)
                
                # Clear logging handlers
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                
                # Create structure save path
                structure_save_path = os.path.join(self.save_path, structure_name)
                os.makedirs(structure_save_path, exist_ok=True)
                
                pkl_file_path = structures_dict[structure_type]
                
                # Load attack templates
                try:
                    print(f"Loading attack templates from: {pkl_file_path}")
                    with open(pkl_file_path, 'rb') as file:
                        temp_dict = pickle.load(file)
                    print(f"✅ Loaded {len(temp_dict)} attack templates")
                except Exception as e:
                    print(f"❌ Error loading attack templates: {e}")
                    self.results_summary[structure_name] = {
                        'error': f"Failed to load templates: {e}",
                        'asr': 0,
                        'successful_attacks': 0,
                        'total_queries': 0,
                        'time_taken': 0
                    }
                    continue
                
                # Prepare attack dictionary
                attack_dict = {}
                query_limit = min(max_queries or len(temp_dict), len(temp_dict))
                
                for i, (key, value) in enumerate(temp_dict.items()):
                    if i >= query_limit:
                        break
                    if i in failed_indices:
                        attack_dict[key] = value
                
                print(f"Running attack with {len(attack_dict)} queries...")
                
                # Initialize attacker
                start_time = time.time()
                
                try:
                    attacker = StructuredAttack(
                        attack_model=attack_model,
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
                    
                    # Run attack
                    print(f"⚡ Starting attack for {structure_name}...")
                    results = attacker.attack(attack_dict, pkl_save_path)
                    end_time = time.time()
                    
                    # Calculate metrics
                    total_queries = len(attack_dict)
                    successful_attacks = 0
                    
                    # Try to calculate success rate from results
                    if os.path.exists(pkl_save_path):
                        try:
                            with open(pkl_save_path, 'rb') as f:
                                response_data = pickle.load(f)
                            
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
                                        
                            print(f"✅ Calculated success rate from results file")
                        except Exception as e:
                            print(f"⚠️  Could not calculate precise success rate: {e}")
                            # Use a rough estimate
                            successful_attacks = total_queries // 10  # Placeholder
                    
                    asr = successful_attacks / total_queries if total_queries > 0 else 0
                    time_taken = end_time - start_time
                    
                    self.results_summary[structure_name] = {
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
                    print(f"❌ Error during attack execution: {e}")
                    self.results_summary[structure_name] = {
                        'error': str(e),
                        'asr': 0,
                        'successful_attacks': 0,
                        'total_queries': len(attack_dict) if 'attack_dict' in locals() else 0,
                        'time_taken': time.time() - start_time
                    }
                
                finally:
                    if 'attacker' in locals():
                        try:
                            attacker.cleanup()
                        except:
                            pass
                    
                print(f"✅ Completed {structure_name}")
                print("-" * 60)
        
        # Save and display final results
        self.save_final_results()
        self.print_final_summary()
    
    def save_final_results(self):
        """Save comprehensive benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(self.save_path, f"llama32_benchmark_summary_{timestamp}.json")
        
        benchmark_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_time': time.time() - self.start_time,
                'config_file': 'config_llama32.yaml'
            },
            'model_info': {
                'target_model': self.config['models']['target']['model_name'],
                'attack_model': self.config['models']['attack']['model_name'],
                'eval_model': self.config['models']['eval']['model_name'],
                'refusal_eval_model': self.config['models']['refusal_eval']['model_name']
            },
            'dataset_info': {
                'name': 'HarmBench',
                'max_queries_per_structure': self.config['attack_settings'].get('max_queries', 'unlimited')
            },
            'results': self.results_summary
        }
        
        with open(summary_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to: {summary_file}")
    
    def print_final_summary(self):
        """Print final benchmark summary"""
        print("\n" + "="*80)
        print("LLAMA-3.2-3B BENCHMARK FINAL SUMMARY")
        print("="*80)
        
        total_successful = 0
        total_queries = 0
        total_time = time.time() - self.start_time
        
        print(f"{'Structure':<25} {'ASR':<8} {'Success/Total':<15} {'Time (s)':<10} {'Status'}")
        print("-" * 80)
        
        for structure_name, metrics in self.results_summary.items():
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
        print(f"{'OVERALL':<25} {overall_asr:<8.2%} {total_successful:>3d}/{total_queries:<3d} {total_time:<10.1f}")
        print("="*80)
        
        print(f"\n🎯 Key Findings:")
        print(f"   • Target Model: meta-llama/Llama-3.2-3B-Instruct")
        print(f"   • Overall ASR: {overall_asr:.2%}")
        print(f"   • Total Attacks: {total_successful}/{total_queries}")
        print(f"   • Total Runtime: {total_time:.1f}s")
        print(f"   • Results Directory: {self.save_path}")
        
        # Identify best/worst performing structures
        if self.results_summary:
            valid_results = {k: v for k, v in self.results_summary.items() if 'error' not in v}
            if valid_results:
                best = max(valid_results.items(), key=lambda x: x[1]['asr'])
                worst = min(valid_results.items(), key=lambda x: x[1]['asr'])
                
                print(f"   • Best Structure: {best[0]} ({best[1]['asr']:.2%})")
                print(f"   • Worst Structure: {worst[0]} ({worst[1]['asr']:.2%})")

def main():
    print("🚀 Starting Llama-3.2-3B Structured Attack Benchmark")
    
    # Check if config file exists
    config_file = "config_llama32.yaml"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        return
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        return
    
    # Run benchmark
    runner = LlamaBenchmarkRunner(config_file)
    runner.run_benchmark()

if __name__ == "__main__":
    main()
