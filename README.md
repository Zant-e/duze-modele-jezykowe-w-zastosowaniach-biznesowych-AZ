# Praca magisterska - Antoni Ziółkowski

## Cel:
Model MoE, gdzie eksperci to adaptery Lora/QLora bazowego modelu Llama 8B. Ograniczenie VRAM - 24GB.

Rozdział 3: Sprawdzenie wydajności Lory i Qlory - różne kwantyzacje, hiperparametry. Znalezienie możliwie najmniejszego zużycia VRAM.\
Rozdział 4: Implementacja wniosków z rozdziału 3 w modelu MoE.\

## Notatki:
Do uruchomienia przykładów/eksperymentów z modelem należy najpierw pobrać model od Mety. \
Bibliotekę lm_eval_harness zainstalowałem manualnie, nie sprawdzałem jeszcze czy wersja z requirements.txt/pipa działa z obecnym kodem. \
Przy aktualnej konfiguracji, skrypty zakładają posiadanie GPU z CUDA i obsługą bfloat16. \

## Already done:
- Implementacja Lory i kwantyzacji bez użycia bibliotek huggingface (zachowanie kontroli i modularności, przydatny skillset do 4 rozdziału). \
- Ładowanie pretrained wag do zmodyfikowanego modelu (funkcjonuje niezależnie od ustawień Lory i kwantyzacji). \
- Implementacja Lory pozwala na konfigurację targetowanych warstw, przy zachowaniu wybranej kwantyzacji dla reszty modelu. \
- Skrypt do ewaluacji - integracja biblioteki lm_eval_harness \
- Porównanie wyników modeli dla różnych kwantyzacji (bez zastosowania LoRy). \
- Skrypt uczący.

## To do:
- [] Wybrać większy dataset
- [] Cupti profiler nie działa poprawnie - naprawić
- [] Integracja Optuny, z opcją monitorowania przez wandb
- [] Hyperparameter tuning
- [] Napisać architekturę MoE