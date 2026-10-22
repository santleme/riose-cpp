"""Animal-group outer folds for the frozen cow activity benchmark."""
from __future__ import annotations

from collections.abc import Sequence


def leave_one_animal_out(animals: Sequence[str]) -> list[dict[str, object]]:
    """Return deterministic indices with one whole animal held out per fold."""
    normalized = [str(animal) for animal in animals]
    animal_ids = sorted(set(normalized))
    if len(animal_ids) < 2:
        raise ValueError("Leave-one-animal-out requires at least two animals")
    folds = []
    for held_out in animal_ids:
        test_indices = [index for index, animal in enumerate(normalized) if animal == held_out]
        train_indices = [index for index, animal in enumerate(normalized) if animal != held_out]
        folds.append({
            "held_out_animal": held_out,
            "train_indices": train_indices,
            "test_indices": test_indices,
        })
    return folds
