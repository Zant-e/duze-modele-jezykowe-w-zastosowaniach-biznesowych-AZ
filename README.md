# Praca magisterska - Antoni Ziółkowski

## General info:
- Repo bazuje na Linuksie, niektóre biblioteki nie działają na Windowsie.
- Do uruchomienia przykładów/eksperymentów z modelem należy najpierw pobrać model od Mety. \
Link: https://llama.meta.com/llama-downloads (do użycia z download.sh)
- Bibliotekę lm_eval_harness zainstalowałem manualnie - wersja z requirements.txt/pipa może nie działać perfekcyjnie.
- Skrypty zakładają posiadanie GPU z CUDA i obsługą bfloat16
- Całość była pisana na maszynie z 24GB VRAM, 64GB RAM - możliwe, że mniejsze zasoby mogą powodować OOM.

## Linki do hyperparam sweepów + wyników uczenia na najlepszych hiperparametrach:
- Non-moe: https://wandb.ai/sghmgr/llama8b_nonmoe_customer
- Moe: https://wandb.ai/sghmgr/llama8b_moe_customer
- Finalne wyniki: https://wandb.ai/sghmgr/llama8b_comparison

## Struktura repozytorium:
### llama3/
- **baseline_eval_results**: Wyniki ewaluacji modelu w zależości od kwantyzacji.
- **download.sh**: Skrypt do pobierania modeli Llama.
- **eval.py**: Skrypt do ewaluacji modeli.
- **example_completion.py**: Przykładowe generowanie tekstu. Możliwość wczytania checkpointu, użycia formatów promptów konwersacyjnych i otwartych.
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