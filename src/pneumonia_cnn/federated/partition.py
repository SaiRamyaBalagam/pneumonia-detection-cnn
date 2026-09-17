"""Non-IID data partitioning for simulated federated learning.

STATUS: not implemented. This module documents the intended interface so a
future session can build the federated learning experiment (the "single
highest-value addition" flagged in the project review) without re-deriving
the design from scratch.

--- The research question ---

Real hospitals don't share data, and their local patient populations differ
(different scanner hardware, different disease prevalence, different
imaging protocols). A federated learning experiment on this project's data
would ask: if we simulate N "hospital" clients, each holding a different
(non-IID) slice of the chest_xray dataset, and train via federated averaging
(e.g. FedAvg) instead of pooling all data centrally -- how much accuracy do
we lose (if any) compared to the centralized model this rebuild already
trains in train.py, and does that gap shrink or grow as data heterogeneity
across clients increases?

--- Intended interface ---

    def partition_non_iid(
        pairs: list[tuple[str, int]],
        n_clients: int,
        alpha: float,
        seed: int,
    ) -> list[list[tuple[str, int]]]:
        '''Split (path, label) pairs across n_clients using a Dirichlet(alpha)
        distribution per class -- the standard non-IID federated learning
        benchmark protocol (Hsu et al. 2019, "Measuring the Effects of
        Non-Identical Data Distribution for Federated Visual Classification").
        Low alpha (e.g. 0.1) => each client is dominated by 1-2 classes
        (extreme non-IID, mimics a hospital that mostly sees pneumonia
        cases). High alpha (e.g. 100) => close to IID, every client's class
        mix resembles the global mix. Sweeping alpha is the experiment.
        '''
        raise NotImplementedError

    def partition_iid(
        pairs: list[tuple[str, int]],
        n_clients: int,
        seed: int,
    ) -> list[list[tuple[str, int]]]:
        '''Baseline: shuffle and split evenly, ignoring label distribution.
        This is the control condition the non-IID runs get compared against.
        '''
        raise NotImplementedError

--- How this plugs into the existing pipeline ---

data.list_pairs() (in ../data.py) already returns exactly the
list[tuple[str, int]] structure these functions would consume -- no changes
needed there. The federated training loop itself would:

  1. Call list_pairs() once to get the full pooled (train+val) pair list,
     same as build_datasets() does today.
  2. Call partition_non_iid(pairs, n_clients, alpha, seed) to get one pair
     list per simulated client.
  3. For each client, build a local tf.data.Dataset the same way
     data._make_tf_dataset() does (that helper is reusable as-is).
  4. Run a FedAvg loop -- Flower (https://flower.ai) is the natural choice
     since it works directly with Keras models and has a simulation mode
     that doesn't need real separate machines. model.build_model() is
     already client-agnostic and would be reused unchanged for each
     client's local model.
  5. Evaluate the federated global model on the same held-out test/ split
     evaluate.py already uses, so results are directly comparable to the
     centralized baseline's numbers.

--- What to measure ---

  - Global test accuracy: federated (per alpha) vs. centralized baseline.
  - Rounds-to-convergence as alpha decreases.
  - Per-client accuracy variance (do some simulated hospitals end up with a
    much worse local model than others under high heterogeneity?).
"""
from __future__ import annotations

from typing import List, Tuple


def partition_non_iid(
    pairs: List[Tuple[str, int]],
    n_clients: int,
    alpha: float,
    seed: int,
) -> List[List[Tuple[str, int]]]:
    raise NotImplementedError(
        "Federated learning is a scaffolded future-work item, not implemented in this pass. "
        "See this module's docstring for the intended Dirichlet-based partitioning design."
    )


def partition_iid(
    pairs: List[Tuple[str, int]],
    n_clients: int,
    seed: int,
) -> List[List[Tuple[str, int]]]:
    raise NotImplementedError(
        "Federated learning is a scaffolded future-work item, not implemented in this pass. "
        "See this module's docstring."
    )
