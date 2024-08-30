# Praca magisterska - Antoni Ziółkowski

## General info:
- Repo bazuje na Linuksie, niektóre biblioteki nie działają na Windowsie.
- Do uruchomienia przykładów/eksperymentów z modelem należy najpierw pobrać model od Mety. \
Link: https://llama.meta.com/llama-downloads (do użycia z download.sh)
- Skrypty zakładają posiadanie GPU z CUDA i obsługą bfloat16
- Całość była pisana na maszynie z 24GB VRAM, 64GB RAM - możliwe, że mniejsze zasoby mogą powodować OOM.


## Struktura repozytorium:
### llama3/
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
E. Hu i in., LoRA: Low-Rank Adaptation of Large Language Models, „ICLR 2022 - 10th International Confer-ence on Learning Representations” (2021), https://arxiv.org/abs/2106.09685v2.