# Praca magisterska - Antoni Ziółkowski

## Cel:
Model MoE, gdzie eksperci to adaptery Lora/QLora bazowego modelu Llama 8B. Ograniczenie VRAM - 24GB.

Rozdział 3: Sprawdzenie wydajności Lory i Qlory - różne kwantyzacje, hiperparametry. Znalezienie możliwie najmniejszego zużycia VRAM.\
Rozdział 4: Implementacja wniosków z rozdziału 3 w modelu MoE.\

## Notatki:
Do uruchomienia przykładów/eksperymentów z modelem należy najpierw pobrać model od Mety.

## Already done:
- Implementacja Lory i kwantyzacji bez użycia bibliotek huggingface (zachowanie kontroli i modularności, przydatny skillset do 4 rozdziału). \
- Ładowanie pretrained wag do zmodyfikowanego modelu (funkcjonuje niezależnie od ustawień Lory i kwantyzacji). \
- Implementacja Lory pozwala na konfigurację targetowanych warstw, przy zachowaniu wybranej kwantyzacji dla reszty modelu. \

## To do:
- [x] Dodać opcję adaptacji Lory dla input embeddingów.
- [x] Sprawdzić czy zamrażać input/output embeddingi i layernormy (niejasne, dodano opcje konfiguracji - do sprawdzenia przy uczeniu)
- [] Dodać opcję inicjalizacji LoftQ
- [] Napisać skrypt eval
- [] Napisać skrypt train