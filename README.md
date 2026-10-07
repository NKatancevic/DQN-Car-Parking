# Autonomous Parking with Dueling Double DQN

Studentski projekat autonomnog parkiranja zasnovan na Deep Q-Learning pristupu.

Agent upravlja automobilom u Pygame parking okruženju i uči da parkira na 8 različitih parking mesta uz prepreke. Finalna verzija koristi **Dueling Double DQN**, experience replay, target network, curriculum learning, teacher guidance i safe-action mask.

## Finalni rezultat

Nezavisna evaluacija: **800 epizoda** (100 po parking mestu).

- Success: **635 / 800 = 79.38%**
- Crash: **0 / 800 = 0.00%**
- Timeout: **165 / 800 = 20.62%**
- 5 od 8 parking mesta: **100% success**

Rezultati po parking mestu:

| Slot | Zona | Success | Crash | Timeout |
|---:|---|---:|---:|---:|
| 0 | LEFT | 66% | 0% | 34% |
| 1 | LEFT | 100% | 0% | 0% |
| 2 | LEFT | 65% | 0% | 35% |
| 3 | RIGHT | 100% | 0% | 0% |
| 4 | RIGHT | 4% | 0% | 96% |
| 5 | RIGHT | 100% | 0% | 0% |
| 6 | TOP | 100% | 0% | 0% |
| 7 | TOP | 100% | 0% | 0% |

## Arhitektura projekta

```text
parking_env_final.py
    Fizika automobila i distance senzori
          ↓
parking_lot_v2.py
    Parking environment, stanje od 31 ulaza, reward i terminalna stanja
          ↓
advanced_dqn_agent.py
    Dueling Double DQN + Replay Buffer + Target Network
          ↓
train_PARKING_LOT_V3_SNAKE_STYLE.py
    Curriculum + teacher + safe-action mask + trening/evaluacija
          ↓
finalni_parking_viewer.py
    Vizuelni prikaz istreniranog modela
```

## State space — 31 ulaz

Agent ne dobija sliku ekrana, već numerički vektor od 31 vrednosti:

- 9 vrednosti: položaj/orijentacija/brzina automobila i odnos prema navigacionom cilju
- 8 distance senzora
- 8 semantičkih senzorskih karakteristika
- 6 geometrijskih karakteristika odnosa automobila i ciljnog parking mesta

## Action space — 8 akcija

0. ništa
1. napred
2. rikverc
3. napred + levo
4. napred + desno
5. rikverc + levo
6. rikverc + desno
7. kočenje

## DQN model

Model koristi **Dueling DQN** arhitekturu.

Shared feature extractor:

```text
31 → 128 → ReLU → 128 → ReLU
```

Zatim se mreža deli na:

```text
Value stream:      128 → 64 → 1
Advantage stream:  128 → 64 → 8
```

Q-vrednosti se dobijaju kombinovanjem value i advantage grane.

Trening koristi i **Double DQN**:
- online mreža bira najbolju narednu akciju
- target mreža procenjuje vrednost izabrane akcije

## Trening strategija

Curriculum learning je podeljen u 6 faza:

1. `TOP6_EMPTY` — jedno mesto bez prepreka
2. `TOP6_OBSTACLES` — isto mesto sa preprekama
3. `TOP_RANDOM` — dva gornja parking mesta
4. `LEFT_RANDOM` — tri leva parking mesta
5. `RIGHT_RANDOM` — tri desna parking mesta
6. `ALL_RANDOM` — svih 8 parking mesta

Dodatno su korišćeni:

- **Teacher guidance** u ranim fazama treninga
- **Safe-action mask** za blokiranje očigledno opasnih poteza
- **Balanced replay**
- **Experience Replay** kapaciteta 100000
- batch size 256
- gamma 0.99
- Adam optimizer
- Huber (`SmoothL1`) loss
- gradient clipping

## Instalacija

Preporučeno: Python 3.11+.

```bash
pip install -r requirements.txt
```

## Pokretanje

### Trening

```bash
python train_PARKING_LOT_V3_SNAKE_STYLE.py
```

Napomena: trenutni trainer je sačuvan u stanju korišćenom za nastavak treninga od Phase 3 i učitava checkpoint `parking_V3_PHASE2_BEST.pth`. Za novi trening od nule potrebno je prilagoditi `START_PHASE` i ukloniti/izmeniti učitavanje checkpointa.

### Finalni viewer

Postavi `parking_V3_FINAL.pth` u root projekta, pa pokreni:

```bash
python finalni_parking_viewer.py
```

Kontrole:

- `1-8` — izbor parking mesta
- `N` — novi random početni položaj
- `R` — ponavljanje istog slučaja
- `A` — random parking mesto
- `SPACE` — pauza
- `S` — prikaz senzora
- `T` — prikaz putanje
- `+ / -` — brzina simulacije
- `ESC` — izlaz

## Glavni fajlovi

- `parking_env_final.py` — model automobila i senzori
- `parking_lot_v2.py` — RL environment
- `advanced_dqn_agent.py` — DQN model, replay buffer i optimizacija
- `train_PARKING_LOT_V3_SNAKE_STYLE.py` — finalni trening pipeline
- `finalni_parking_viewer.py` — demonstracija istreniranog agenta

## Tehnologije

- Python
- PyTorch
- Pygame
- NumPy

## Napomena

Model checkpoint (`.pth`) nije obavezan za pregled izvornog koda. Ako se postavlja na GitHub, zbog veličine fajla može biti pogodnije koristiti Git LFS ili GitHub Release.
