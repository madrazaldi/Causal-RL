from __future__ import annotations

import argparse

from .parallel import resolve_n_jobs
from .policy_learning import main_results_placeholder, save_policy_artifacts, train_all_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior, heuristic, and fitted-Q policies.")
    parser.add_argument("--n-jobs", default=None, help="Parallel workers: auto, -1, or a positive integer.")
    args = parser.parse_args()

    n_jobs = resolve_n_jobs(args.n_jobs)
    artifacts = train_all_policies(n_jobs=n_jobs)
    save_policy_artifacts(artifacts)
    main_results_placeholder()
    print(f"Trained behavior model, heuristic baseline, and FQI policies with n_jobs={n_jobs}.")


if __name__ == "__main__":
    main()
