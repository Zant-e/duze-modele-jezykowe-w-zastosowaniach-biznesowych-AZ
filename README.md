# Praca magisterska - Antoni Ziółkowski

## Ogólne informacje:
- Repozytorium bazuje na Linuksie, niektóre biblioteki nie działają na Windowsie.
- Do uruchomienia przykładów/eksperymentów z modelem należy najpierw pobrać model od Mety. \
Link: https://llama.meta.com/llama-downloads (do użycia z download.sh)
- Skrypty zakładają posiadanie GPU z CUDA i obsługą bfloat16
- Całość była pisana na maszynie z 24GB VRAM, 64GB RAM - możliwe, że mniejsze zasoby mogą powodować OOM.


## Struktura repozytorium:
### llama3/
- **results/**: Wyniki eksperymentów.
    - **lm_harness_eval_results**: Wyniki ewaluacji na benchmarkach, w tym ewaluacja katastroficznego zapominania dostrojonych modeli.
    - **ft_results**: Wyniki dostrajania modeli, w tym hiperparametry, przebieg uczenia, przykłady generowanego tekstu.
- **download.sh**: Skrypt do pobierania modeli Llama.
- **eval_pre_tune.py**: Skrypt do ewaluacji modeli przed dostrajaniem.
- **eval_post_tune.py**: Skrypt do ewaluacji modeli po dostrajaniu.
- **memory_usage_per_q.ipynb**: Analiza zużycia pamięci w zależności od kwantyzacji.
- **text_generation.py**: Generowanie tekstu.
- **finetune.py**: Skrypt do dostrajania modeli (argumenty można zmieniać w utils/configs, zamiast podawać je w CLI).
- **llama/**: Główny kod modelu Llama.
    - **generation.py**: Utils - inicjalizacja, ładowanie checkpointów, ogólna funkcja do generowania tekstu, przygotowanie warstw do uczenia.
    - **model.py**: Implementacja modelu.
    - **tokenizer.py**: Implementacja tokenizera.
- **Meta-Llama-3-8B**: Pliki modelu 8B pretrained (wagi nie uploadowane na git).
- **Meta-Llama-3-8B-Instruct**: Pliki modelu 8B instruct (wagi nie uploadowane na git).
- **tuned_checkpoints**: Wagi wytrenowanych modeli (LoRA + gate, bazowe wagi nadal wymagane do uruchomienia).
- **sweep_config.yaml**: Plik konfiguracyjny dla Wandb hyperparam sweepów.
- **utils/**: Skrypty i funkcje pomocnicze.
  - **configs.py**: Konfiguracja modelu i wandb do skryptu finetune.
  - **dataset_utils.py**: Procesowanie datasetów.
  - **eval_utils.py**: Wrappery modelu, aby współpracował z LM eval harness.
  - **memory_utils.py**: Monitorowanie pamięci.
  - **train_utils.py**: Główna pętla ucząca, customowe funkcje straty.

### Licences: Licencje


## Źródła:
- E. Hu i in., LoRA: Low-Rank Adaptation of Large Language Models, „ICLR 2022 - 10th International Confer-ence on Learning Representations” (2021), https://arxiv.org/abs/2106.09685v2.
- Meta-llama/llama-recipes: Scripts for fine-tuning Meta Llama3 [na:] https://github.com/meta-llama/llama-recipes, dostęp 27 sierpnia 2024 r.
- Meta-llama/llama3: The official Meta Llama 3 GitHub site [na:] https://github.com/meta-llama/llama3, dostęp 14 lipca 2024 r.
- Jiang A.Q., Sablayrolles A., Roux A., i in., Mixtral of Experts, (2024), https://arxiv.org/abs/2401.04088v1.
- Fedus W., Zoph B., Shazeer N., Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, „Journal of Machine Learning Research” t. 23 (2021), https://arxiv.org/abs/2101.03961v3.
- EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models. [na:] https://github.com/EleutherAI/lm-evaluation-harness, dostęp 27 sierpnia 2024 r.
- Dettmers T., Pagnoni A., Holtzman A., Zettlemoyer L., QLoRA: Efficient Finetuning of Quan-tized LLMs, „Advances in Neural Information Processing Systems” t. 36 (2023), https://arxiv.org/abs/2305.14314v1.
- Bitext/Bitext-customer-support-llm-chatbot-training-dataset · Datasets at Hugging Face [na:] https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset, dostęp 27 sierpnia 2024 r.
- LDJnr/Pure-Dove · Datasets at Hugging Face [na:] https://huggingface.co/datasets/LDJnr/Pure-Dove, dostęp 27 sierpnia 2024 r.
