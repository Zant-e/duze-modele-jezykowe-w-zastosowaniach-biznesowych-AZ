import lm_eval
from llama import Llama
from utils.eval_utils import HarnessModel
import json
import os
import gc
import torch
import time


def main(path_list, limit=100):

    for path in path_list:
        json_config = json.load(open(f"{path}/model_config.json", "r"))

        model_name = path.split("/")[-1]
        print(f"Evaluating model: {model_name}")

        model_config = {
            "model_type": json_config["model_type"],
            "quant_type": json_config["quant_type"],
            "lora_ckpt_path": f"{path}/model.pt",
            "lora_target": json_config["lora_target"],
            "lora_r": json_config["lora_r"],
            "lora_alpha": json_config["lora_alpha"],
            "lora_dropout": json_config["lora_dropout"],
            "context_length": json_config["context_length"],
            "seed": 1,
            "use_moe": json_config["use_moe"],
            "num_experts": json_config["num_experts"],
            "num_experts_per_tok": json_config["num_experts_per_tok"],
            "max_batch_size": 1,
            "use_cache": True,
        }

        generator = Llama.build(
            instruct_model=model_config["model_type"] == "instruct",
            max_seq_len=model_config["context_length"],
            max_batch_size=model_config["max_batch_size"],
            seed=model_config["seed"],
            lora_target=model_config["lora_target"],
            lora_r=model_config["lora_r"],
            lora_alpha=model_config["lora_alpha"],
            lora_dropout=model_config["lora_dropout"],
            quant_type=model_config["quant_type"],
            use_moe=model_config["use_moe"],
            num_experts=model_config["num_experts"],
            num_experts_per_tok=model_config["num_experts_per_tok"],
            use_cache=model_config["use_cache"],
            lora_ckpt_path=model_config["lora_ckpt_path"],
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
            # "lambada": 2,
            "prost": None,
        }
        results = {}

        start_time = time.time()

        for task, num_fewshot in task_dict.items():
            task_result = lm_eval.simple_evaluate(
                model=mymodel,
                tasks=[task],
                num_fewshot=num_fewshot,
                batch_size=model_config["max_batch_size"],
                limit=limit,
            )
            results[task] = task_result["results"]

        file_path = f"ft_eval_results/{model_name}.json"

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

    path_list = [
        # "tuned_checkpoints/customer_dataset/llama3_8b_instruct_customer",
        # "tuned_checkpoints/customer_dataset/llama3_8b_moe_instruct_customer",
        # "tuned_checkpoints/customer_dataset/llama3_8b_pretrained_customer",
        "tuned_checkpoints/customer_dataset/llama3_8b_moe_pretrained_customer",
        "tuned_checkpoints/mixed_dataset/llama3_8b_instruct_mixed",
        "tuned_checkpoints/mixed_dataset/llama3_8b_moe_instruct_mixed",
        "tuned_checkpoints/mixed_dataset/llama3_8b_pretrained_mixed",
        "tuned_checkpoints/mixed_dataset/llama3_8b_moe_pretrained_mixed",
    ]

    main(path_list)
