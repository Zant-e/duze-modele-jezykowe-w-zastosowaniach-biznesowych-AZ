import os
import time
import yaml
from pathlib import Path
from datetime import datetime
import contextlib

import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import loralib as lora

from utils.memory_utils import MemoryTrace


# Custom loss function - shifts tensors, masks out unwanted tokens
def cross_entropy_loss(logits, labels, loss_mask):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    raw_loss = F.cross_entropy(logits, labels, reduction="none")
    loss_mask = loss_mask[:, 1:].contiguous().view(-1)
    loss = (raw_loss * loss_mask).sum() / loss_mask.sum()

    return loss


@contextlib.contextmanager
def profile(cfg):
    use_profiler: bool = cfg.use_profiler
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(
                f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            )
        print(
            f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}"
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait_step, warmup=warmup_step, active=active_step, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    wandb_run=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloader: The dataloader containing the eval data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        train_config: The training configuration

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
            with profile(train_config) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if (
                        train_config.max_train_step > 0
                        and total_train_steps > train_config.max_train_step
                    ):
                        max_steps_reached = True
                        print(
                            "max training steps reached, stopping training, total train steps finished: ",
                            total_train_steps - 1,
                        )
                        break
                    for key in batch.keys():
                        batch[key] = batch[key].to("cuda:0")

                    logits = model.forward(batch["input_ids"])
                    loss = cross_entropy_loss(
                        logits, batch["input_ids"], batch["loss_mask"]
                    )
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(
                            float(torch.exp(loss.detach().float()))
                        )
                    total_loss += loss.detach().float()

                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        if (
                            train_config.gradient_clipping
                            and train_config.gradient_clipping_threshold > 0.0
                        ):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                train_config.gradient_clipping_threshold,
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                    if train_config.use_profiler:
                        profile_context.step()
                    if wandb_run:
                        wandb_run.log(
                            {
                                "train/epoch": epoch + 1,
                                "train/step": epoch * len(train_dataloader) + step,
                                "train/loss": loss.detach().float(),
                            }
                        )

                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                    )

                    if train_config.save_metrics:
                        save_to_json(
                            metrics_filename,
                            train_step_loss,
                            train_loss,
                            train_step_perplexity,
                            train_prep,
                            val_step_loss,
                            val_loss,
                            val_step_perplexity,
                            val_prep,
                        )
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / len(train_dataloader)

        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model, train_config, eval_dataloader, wandb_run
            )
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()

            # Save the model checkpoint if the validation loss is the best so far
            save_file = f"{train_config.output_dir}/{train_config.model_name}.pt"
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                torch.save(lora.lora_state_dict(model), save_file)
                print(f"PEFT modules are saved in {train_config.output_dir} directory")

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        print(
            f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
        )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    return results


def evaluation(model, train_config, eval_dataloader, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        train_config: The training configuration
        eval_dataloader: The dataloader containing the evaluation data
        wandb_run: Flag to indicate if wandb is being used

    Returns: eval_ppl, eval_epoch_loss
    """

    model.eval()
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(
            tqdm(
                eval_dataloader,
                colour="green",
                desc="evaluating Epoch",
                dynamic_ncols=True,
            )
        ):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if (
                train_config.max_eval_step > 0
                and total_eval_steps > train_config.max_eval_step
            ):
                print(
                    "max eval steps reached, stopping evaluation, total_eval_steps: ",
                    total_eval_steps - 1,
                )
                break
            for key in batch.keys():
                batch[key] = batch[key].to("cuda:0")
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                logits = model.forward(batch["input_ids"])
                loss = cross_entropy_loss(
                    logits, batch["input_ids"], batch["loss_mask"]
                )

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log(
            {
                "eval/perplexity": eval_ppl,
                "eval/loss": eval_epoch_loss,
            },
            commit=False,
        )

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def save_train_params(train_config):
    """
    This function saves the train_config into a train_params.yaml.
    """
    # Convert the train_config object to a dictionary,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {
        k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")
    }

    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict}
    # Construct the folder name using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, "train_params.yaml")

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, "w") as f:
            f.write(config_yaml)
        print(f"training params are saved in {file_name}")


def save_to_json(
    output_filename,
    train_step_loss,
    train_epoch_loss,
    train_step_ppl,
    train_epoch_ppl,
    val_step_loss,
    val_epoch_loss,
    val_step_ppl,
    val_epoch_ppl,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
