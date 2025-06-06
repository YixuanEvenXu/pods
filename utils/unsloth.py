"""
This file is adapted from unsloth.models.rl (unsloth version 2025.2.4)

Adapted by: Yixuan Even Xu in 2025
"""

__all__ = [
    "PatchFastRL",
]

METRICS_MOVE_TO_END = [
    "nll",
    "aux",
    "beta",
    "alpha",
]
import torch
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import inspect
import os
import re
import functools
from unsloth_zoo.compiler import create_new_function


def PatchRL(FastLanguageModel):

    from trl.models.utils import unwrap_model_for_generation
    from contextlib import contextmanager

    @contextmanager
    def unsloth_unwrap_model_for_generation(model, *args, **kwargs):
        with unwrap_model_for_generation(model, *args, **kwargs) as unwrapped_model:
            # Put the model in inference mode.
            FastLanguageModel.for_inference(unwrapped_model)

            # We must use .clone for Unsloth since we force inference_mode
            # Rather we should have used no_grad
            original_generate = unwrapped_model.generate
            def generate_with_clone(*args, **kwargs):
                out = original_generate(*args, **kwargs)
                if isinstance(out, torch.Tensor):
                    return out.clone()
                return out
            pass
            unwrapped_model.generate = generate_with_clone

            try:
                yield unwrapped_model
            finally:
                # Restore generate and return
                unwrapped_model.generate = original_generate
                FastLanguageModel.for_training(model)
            pass
        pass
    pass

    import trainer
    unwrap = "unwrap_model_for_generation"
    trainers = dir(trainer)
    trainers = [x for x in trainers if x.endswith("_trainer")]
    for specific_trainer in trainers:
        if hasattr(eval(f"trainer.{specific_trainer}"), unwrap):
            exec(f"trainer.{specific_trainer}.{unwrap} = unsloth_{unwrap}")
    pass
pass


def _patch_trl_rl_trainers(trainer_file = "treepo_trainer"):
    # Patch for vLLM and Unsloth PEFT
    
    import trainer
    specific_trainer = eval(f"trainer.{trainer_file}")
    name = [x for x in dir(specific_trainer) if x.endswith("Trainer") and x != "Trainer" and trainer_file.split("_")[0] in x.lower()]
    assert(len(name) == 1)
    RLTrainer_name = name[0]
    print(f"trainer.{trainer_file}.{RLTrainer_name}")
    RLTrainer = eval(f"trainer.{trainer_file}.{RLTrainer_name}")

    try:
        __init__ = inspect.getsource(RLTrainer.__init__)
    except:
        # Already patched most likely!
        return
    old__init__ = __init__
    all_imports = dir(specific_trainer)
    assert("Union" in all_imports)
    imports = [x for x in all_imports if not x.startswith("_")]
    imports += ["Trainer"]

    spaces = __init__.find("def")
    __init__ = __init__.split("\n")
    __init__ = "\n".join(x[spaces:] for x in __init__)

    # Replace vLLM sections since we already have it done!
    vllm_part = re.findall(
        r"(\n[\s]{4}"\
        r"if (self|args)\.use_vllm\:.+?"\
        r"\n[\s]{4,}"\
        "else:\n)",
        __init__,
        flags = re.MULTILINE | re.DOTALL,
    )
    if (len(vllm_part) != 1): return

    vllm_part, args = vllm_part[0][0], vllm_part[0][1]
    # Strip all comments
    new_vllm_part = re.sub(r"\#[^\n]{1,}\n", "", vllm_part)

    # Get SamplingParams
    sampling_params = re.findall(
        r"\n[\s]{4,}(self\.[^\s]{1,}[\s]{0,}\=[\s]{0,}"\
        r"SamplingParams\(.+?\))",
        new_vllm_part,
        flags = re.MULTILINE | re.DOTALL,
    )
    if len(sampling_params) != 1: return

    sampling_params = sampling_params[0]
    # Replace with our vLLM engine
    sampling_params = \
        " "*8 + "self.llm = model.vllm_engine; self._last_loaded_step = 0; " + \
        sampling_params # Add spaces
    new_vllm_part = f"\n    if {args}.use_vllm:\n{sampling_params}\n    else:\n"
    __init__ = __init__.replace(vllm_part, new_vllm_part)

    # Remove peft_config
    __init__ = __init__.replace("elif peft_config is None:", "elif False:")
    __init__ = __init__.replace("elif peft_config is not None:", "elif False:")
    __init__ = __init__.replace("if peft_config is None:", "if False:")
    __init__ = __init__.replace("if peft_config is not None:", "if False:")
    __init__ = __init__.replace("get_peft_model(model, peft_config)", "model")

    # Add spaces back into __init__
    __init__ = __init__.split("\n")
    __init__ = "\n".join(' '*spaces + x for x in __init__)

    # Search for vLLM calling in all child functions
    functions = dir(RLTrainer)
    RLTrainer_source = inspect.getsource(RLTrainer)
    functions = [x for x in functions if f"def {x}" in RLTrainer_source]

    changed = {"__init__" : (old__init__, __init__,)}
    for function in functions:
        if not hasattr(RLTrainer, function): continue
        fx = getattr(RLTrainer, function)
        try:
            source = inspect.getsource(fx)
        except:
            continue
        original_source = source

        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        source = re.sub(
            r"(\n[\s]{4,}).+?model_executor\.driver_worker.+?\n",
            r"\n\1pass\n",
            source,
        )

        # llm_model.load_weights(model.state_dict().items())
        source = re.sub(
            r"(\n[\s]{4,}).+?load_weights\(.+?\n",
            r"\n\1pass\n",
            source,
        )

        # .state_dict()
        source = re.sub(
            r"\.state_dict\(\)",
            r"",
            source,
        )
        
        # Replace self.llm.generate and self.llm.chat
        lora_name = trainer_file + "_lora_model"
        source = re.sub(
            r"(self\.llm\.(?:generate|chat)\([^\)]{1,})\)",
            r"\1, lora_request = self.model.load_lora('" + lora_name + r"', load_tensors = True))",
            source
        )

        # Skip if no changes done
        if source == original_source: continue

        # Find all imports
        imports += [x for x in all_imports if not x.startswith("_") and x in source]

        changed[function] = (original_source, source,)
    pass

    # Import all functions
    imports = list(set(imports))

    # Patch all functions
    for function in changed:
        old, new = changed[function]
        RLTrainer_source = RLTrainer_source.replace(old, new)
    pass
    RLTrainer_source = RLTrainer_source.replace(
        f"class {RLTrainer_name}", f"class Unsloth{RLTrainer_name}", 1
    )

    # Create new class in compiled cache and import it
    module = create_new_function(
        RLTrainer_name,
        RLTrainer_source,
        f"trainer.{trainer_file}",
        imports,
    )

    # Patch over modules
    exec(f"{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trainer.{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    exec(f"trainer.{trainer_file}.{RLTrainer_name} = module.Unsloth{RLTrainer_name}", locals(), globals())
    return module
pass


def patch_trl_rl_trainers():
    import trainer
    all_trainers = dir(trainer)
    all_trainers = [x for x in all_trainers if x.islower() and x.endswith("_trainer")]
    for specific_trainer in all_trainers:
        _patch_trl_rl_trainers(specific_trainer)
    return
pass


def PatchFastRL(algorithm = "TreePO", FastLanguageModel = None):
    if FastLanguageModel is not None: PatchRL(FastLanguageModel)
    patch_trl_rl_trainers()
pass
