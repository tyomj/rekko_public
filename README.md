# Rekko challenge 2019

### Quick start

1. Git clone this repo
2. Install an instance of Ocean


   `make -B package`

3. Place raw data into `/data/raw/` directory
4. Go to the experiment folder `exp-002-Implicit_recency`
5. Use one of the following commands to run one of the experiments respectively:
    - `make -B search_ab`
    - `make -B search_pruning`

    or just review `./scripts/train_and_search.py`