from easyjailbreak.mutation import MutationBase
from easyjailbreak.datasets import Instance
import re
import json
import random
import pdb
from typing import List
import importlib.resources

class WildbenchMutator(MutationBase):
    """
    The 'WildbenchMutator' class, inheriting from 'MutationBase', is designed to mutate
    a prompt with tactics for the Wildbench dataset. 
    """

    def __init__(self, model, base_prompt, attr_name='jailbreak_prompt', schema_type='json'):
        """
        Initializes the SchemaMutator instance.
        :param ~ModelBase model: The model to be used for text generation and schema mutation.
        :param str attr_name: The attribute name in the instance to be altered.
        :param str schema_type: The type of schema being used (e.g., 'json', 'sql').
        """
        self.model = model
        self.attr_name = attr_name
        self.schema_type = schema_type
        self.tactics = self._load_tactics()
        # set main_tactics to the tactics that have keys to do with schema_type and the one 'templated output format'. Give me the keys as a list
        self.main_tactics = [k for k in self.tactics.keys() if schema_type.lower() in k.lower() or 'templated output format' in k.lower()]
        # from self.tactics which is a dict, remove the tactics that have keys to do with 'json', 'csv' and 'sql'
        # and remove the tactics that have keys to do with 'templated output format'
        # just need the keys as lists
        self.mutation_tactics = [k for k in self.tactics.keys() if 'json' not in k.lower() and 'csv' not in k.lower() and 'sql' not in k.lower() and 'templated output format' not in k.lower()]
        self.num_tactics_per_attack = 5
        self.num_excerpts_per_tactic = 5
        self.base_prompt = base_prompt

    def _convert_tactics_list_to_string(self, tactics, tactics_definition, tactics_examples):
        """
        Converts the tactics list to a formatted string.
        """
        tactics_string = ""

        for t, d, es in zip(tactics, tactics_definition, tactics_examples):
            tactics_string += f"- {t}: [definition] {d} [example] "
            tactics_string += " [example] ".join(es)
            tactics_string += "\n"
        return tactics_string

    def _load_tactics(self) -> dict:
        """
        Loads mutation tactics from a JSON file.
        """
        tactics_path = importlib.resources.files("easyjailbreak.seed") / "wildbench_tactics_subset.json"
        with open(tactics_path, 'r') as f:
            tactics = json.load(f)

        to_exclude_tactics = ["strategy name", "name of the first new strategy", ""]
        does_not_contain = ["explicit", "sex", "erotic"]
        tactics = {k: v for k, v in tactics.items()
                                if (k not in to_exclude_tactics)
                                and k[-1] != ":"
                                and (all(x not in v["definition"][0].lower() for x in does_not_contain))
                                and (all(x not in k.lower() for x in does_not_contain))}
        return tactics

    def format_attack_single(self, behavior, tactics, target_str=None):
        """
        Format the input string for a single attack.
        """
        tactics_definition = []
        tactics_examples = []
        for t in tactics:
            # To ensure enough excerpts exist for tactic t
            num_excerpts = min(self.num_excerpts_per_tactic, self.tactics[t]["excerpt"])

            tactics_definition.extend(random.sample(self.tactics[t]["definition"], 1))
            tactics_examples.append(random.sample(self.tactics[t]["excerpt"], num_excerpts))
        formatted_strategies_list = self._convert_tactics_list_to_string(tactics, tactics_definition, tactics_examples)

        compose_prompt = self.base_prompt.replace("[INSERT STRATEGY LIST]", formatted_strategies_list)
        compose_prompt = compose_prompt.replace("[INSERT SIMPLE PROMPT]", behavior)
        if target_str is not None:
            compose_prompt = compose_prompt.replace("[INSERT STARTING STRING]", target_str)

        return compose_prompt

    def _select_tactics(self):
        """
        Selects jailbreak tactics for generating the attack based on different selection methods.
        """

        # Randomly select tactics from self.mutation_tactics list
        return random.sample(self.mutation_tactics, self.num_tactics_per_attack)

        
    def mutate_schema(self, query: str) -> str:
        selected_tactics = self._select_tactics()
        # combine selected tactics and self.main_tactics
        tactics = selected_tactics + self.main_tactics
        prompt = self.format_attack_single(query, selected_tactics, target_str=None)
        mutated_schema = self.model.generate(prompt)
        
        return mutated_schema

    def _get_mutated_instance(self, instance: Instance) -> List[Instance]:
        schema = getattr(instance, self.attr_name)
        mutated_schema = self.mutate_schema(schema)
        
        new_instance = instance.copy()
        setattr(new_instance, self.attr_name, mutated_schema)
        new_instance.parents = [instance]
        new_instance.children = []
        
        return [new_instance]
