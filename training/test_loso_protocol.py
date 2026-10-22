import unittest

try:
    from .loso_protocol import leave_one_animal_out
except ImportError:  # pragma: no cover - direct test discovery from training/
    from loso_protocol import leave_one_animal_out


class LeaveOneAnimalOutProtocolTest(unittest.TestCase):
    def test_each_animal_is_wholly_held_out_once(self):
        animals = ["cow-a", "cow-b", "cow-a", "cow-c", "cow-b"]
        folds = leave_one_animal_out(animals)
        self.assertEqual([fold["held_out_animal"] for fold in folds], ["cow-a", "cow-b", "cow-c"])
        for fold in folds:
            held_out = fold["held_out_animal"]
            test_ids = {animals[i] for i in fold["test_indices"]}
            train_ids = {animals[i] for i in fold["train_indices"]}
            self.assertEqual(test_ids, {held_out})
            self.assertNotIn(held_out, train_ids)
            self.assertEqual(len(fold["train_indices"]) + len(fold["test_indices"]), len(animals))

    def test_requires_more_than_one_animal(self):
        with self.assertRaises(ValueError):
            leave_one_animal_out(["cow-a"])


if __name__ == "__main__":
    unittest.main()
