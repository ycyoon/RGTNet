"""
Jailbroken Class
============================================
Jailbroken utilized competing objectives and mismatched generalization 
modes of LLMs to constructed 29 artificial jailbreak methods.

Paper title: Jailbroken: How Does LLM Safety Training Fail?
arXiv Link: https://arxiv.org/pdf/2307.02483.pdf
"""

import logging
import os
import pickle
from typing import List, Tuple
logging.basicConfig(level=logging.INFO)
from easyjailbreak.metrics.Evaluator import EvaluatorGenerativeJudge, EvaluatorHarmBench
from easyjailbreak.metrics.Evaluator.Evaluator_RefusalJudge import EvaluatorRefusalJudge
from easyjailbreak.attacker import AttackerBase
from easyjailbreak.datasets import JailbreakDataset, Instance
from easyjailbreak.mutation.rule import *

__all__ = ['Jailbroken']
     
class Jailbroken(AttackerBase):
    r"""
    Implementation of Jailbroken Jailbreak Challenges in Large Language Models
    """
    def __init__(self, attack_model, target_model, eval_model, refusal_eval_model, jailbreak_datasets: JailbreakDataset, rounds: int = 10, 
                 save_path: str = 'cached_instances', 
                 use_cache: bool = True):
        r"""
        :param attack_model: The attack_model is used to generate the adversarial prompt.
        :param target_model: The target language model to be attacked.
        :param eval_model: The evaluation model to evaluate the attack results.
        :param jailbreak_datasets: The dataset to be attacked.
        :param template_file: The file path of the template.
        """
        super().__init__(attack_model, target_model, eval_model, jailbreak_datasets)
        self.mutations = [
            Artificial(attr_name='query'),
            Base64(attr_name='query'),
            Base64_input_only(attr_name='query'),
            Base64_raw(attr_name='query'),
            Disemvowel(attr_name='query'),
            Leetspeak(attr_name='query'),
            Rot13(attr_name='query'),
            Combination_1(attr_name='query'),
            Combination_2(attr_name='query'),
            Combination_3(attr_name='query'),
            Auto_payload_splitting(self.attack_model,attr_name='query'),
            Auto_obfuscation(self.attack_model,attr_name='query'),

        ]
        # New attributes
        self.evaluator = EvaluatorHarmBench(eval_model)
        self.refusal_evaluator = EvaluatorRefusalJudge(refusal_eval_model)
        self.total_queries = 0
        self.successful_queries = 0
        self.total_attempts = 0
        self.successful_attempts = 0
        self.attacks_until_success = []
        self.rounds = rounds
        self.save_path = save_path
        self.use_cache = use_cache
        os.makedirs(self.save_path, exist_ok=True)

    def get_cached_instances(self, instance: Instance) -> List[Instance]:
        if not self.use_cache:
            return None
        cache_file = os.path.join(self.save_path, f"{instance.id}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def cache_instances(self, instance: Instance, transformed_instances: List[Instance]):
        if not self.use_cache:
            return
        cache_file = os.path.join(self.save_path, f"{instance.id}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(transformed_instances, f)

    def generate_attack_prompts(self):
        for instance in self.jailbreak_datasets:
            cached_instances = self.get_cached_instances(instance)
            if cached_instances is None:
                source_instance_list = []
                for mutation in self.mutations:
                    transformed_jailbreak_datasets = mutation(JailbreakDataset([instance]))
                    for item in transformed_jailbreak_datasets:
                        source_instance_list.append(item)
                self.cache_instances(instance, source_instance_list)
                logging.info(f"Generated and cached attack prompts for instance {instance.id}")
            else:
                logging.info(f"Loaded cached attack prompts for instance {instance.id}")

    def single_attack(self, instance: Instance) -> Tuple[JailbreakDataset, int, bool]:
        source_instance_list = self.get_cached_instances(instance)
        updated_instance_list = []
        attempts = 0
        success_found = False

        if source_instance_list is None:
            source_instance_list = []
            for mutation in self.mutations:
                transformed_jailbreak_datasets = mutation(JailbreakDataset([instance]))
                for item in transformed_jailbreak_datasets:
                    source_instance_list.append(item)
            self.cache_instances(instance, source_instance_list)

        # Ensure we have at least self.rounds instances
        while len(source_instance_list) < self.rounds:
            source_instance_list.extend(source_instance_list[:self.rounds - len(source_instance_list)])
        
        for instance in source_instance_list[:]:
            attempts += 1
            answer = self.target_model.generate(instance.jailbreak_prompt.format(query = instance.query))
            instance.target_responses.append(answer)
            updated_instance_list.append(instance)
            
            # Evaluate after each attempt
            self.evaluator([instance])
            self.refusal_evaluator([instance])
            if instance.eval_results[0] == 1 and not success_found:
                self.attacks_until_success.append(attempts)
                success_found = True

        return JailbreakDataset(updated_instance_list), attempts, success_found

    def attack(self):
        logging.info("Jailbreak started!")
        self.attack_results = JailbreakDataset([])
        total_refusals = 0
        total_attempts_refusal = 0
        try:
            for instance in self.jailbreak_datasets:
                results, attempts, success = self.single_attack(instance)
                self.total_queries += 1
                self.total_attempts += attempts
                
                for new_instance in results:
                    self.attack_results.add(new_instance)
                    total_refusals += new_instance.eval_results[1]
                    total_attempts_refusal += 1
                if success:
                    self.successful_queries += 1
                    self.successful_attempts += 1
        except KeyboardInterrupt:
            logging.info("Jailbreak interrupted by user!")
        
        self.refusal_rate = total_refusals / total_attempts_refusal
        self.update(self.attack_results)
        logging.info("Jailbreak finished!")

    def update(self, Dataset: JailbreakDataset):
        r"""
        Update the state of the Jailbroken based on the evaluation results of Datasets.
        
        :param Dataset: The Dataset that is attacked.
        """
        # for prompt_node in Dataset:
        #     self.current_jailbreak += prompt_node.num_jailbreak
        #     self.current_query += prompt_node.num_query
        #     self.current_reject += prompt_node.num_reject

        # Calculate new metrics
        self.asr_query = self.successful_queries / self.total_queries if self.total_queries > 0 else 0
        self.asr_total = self.successful_attempts / self.total_attempts if self.total_attempts > 0 else 0
        self.efficiency = sum(self.attacks_until_success) / len(self.attacks_until_success) if self.attacks_until_success else 0

    def log(self):
        with open(os.path.join(self.save_path, "jailbreak_report.txt"), "w") as f:
            f.write("======Jailbreak report:======\n")
            f.write(f"Total instances: {self.total_queries}\n")
            f.write(f"Successful instances: {self.successful_queries}\n")
            f.write(f"Total attempts: {self.total_attempts}\n")
            f.write(f"Successful attempts: {self.successful_attempts}\n")
            instance_asr = self.successful_queries / self.total_queries if self.total_queries > 0 else 0
            f.write(f"Instance-level ASR: {instance_asr:.2%}\n")
            f.write(f"ASR Query: {self.asr_query:.2%}\n")
            f.write(f"ASR Total: {self.asr_total:.2%}\n")
            f.write(f"Efficiency (avg queries until success): {self.efficiency:.2f}\n")
            f.write(f"Refusal Rate: {self.refusal_rate:.2%}\n")
            f.write("========Report End===========\n")
        
        logging.info("Jailbreak report saved to jailbreak_report.txt")
