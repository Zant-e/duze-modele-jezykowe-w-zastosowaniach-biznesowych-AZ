import lm_eval
from llama import Llama
from utils.eval_utils import HarnessModel
import json
import os
import gc
import torch
import time


def main(model_type, quant_type, limit=100):

    generator = Llama.build(
        instruct_model=model_type == "instruct",
        max_seq_len=8192,
        max_batch_size=1,
        seed=1,
        lora_target=[],
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        quant_type=quant_type,
    )

    mymodel = HarnessModel(generator)

    task_dict = {
        "arc_challenge": 25,
        "hellaswag": 10,
        "truthfulqa_mc1": 0,
        "winogrande": 5,
        "gsm8k": 5,
        "mmlu": 5,
        "glue": 2,
        "super-glue-lm-eval-v1": 2,
        "lambada": 2,
        "prost": None,
    }
    results = {}

    start_time = time.time()

    for task, num_fewshot in task_dict.items():
        task_result = lm_eval.simple_evaluate(
            model=mymodel,
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=1,
            limit=limit,
        )
        results[task] = task_result["results"]

    file_path = f"baseline_eval_results/results_llama_8b_{model_type}_{quant_type}_limit_{limit}.json"

    # Load existing results if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Merge new results with existing results
    for key, value in results.items():
        if key in existing_results:
            existing_results[key].update(value)
        else:
            existing_results[key] = value

    # Save the merged results back to the JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4)

    # Deleting model and variables to free up VRAM
    generator.model.cpu()
    del (
        generator.model,
        generator.tokenizer,
        generator,
        mymodel,
        task_dict,
        results,
        existing_results,
    )
    gc.collect()
    torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken: {int(minutes)}:{seconds:04.1f}")


if __name__ == "__main__":
    model_types = ["instruct", "pretrained"]
    quant_types = ["nf4", "fp4", "8bit", ""]

    for model_type in model_types:
        for quant_type in quant_types:
            print(f"Running for model_type: {model_type}, quant_type: {quant_type}")
            main(model_type, quant_type)
