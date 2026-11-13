"""Regression tests for the public Python API documented by README.md.

These tests intentionally exercise the public package import and numerical behavior,
not implementation details alone. They should run with ``python -m unittest`` after
installing the package.
"""

import unittest

import velograd as vg
import velograd.nn as nn
import velograd.optim as optim


class PythonApiParityTests(unittest.TestCase):
    def test_readme_training_example_apis_are_available(self):
        model = nn.Sequential(
            nn.Linear(3, 4),
            nn.RMSNorm(4),
            nn.SiLU(),
            nn.Linear(4, 2),
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.assertTrue(model.parameters())
        self.assertTrue(callable(optimizer.step))
        self.assertTrue(callable(nn.CrossEntropyLoss().forward))

    def test_binary_broadcast_uses_aligned_dimensions(self):
        lhs = vg.tensor([[1.0], [2.0]])
        rhs = vg.tensor([[10.0, 20.0, 30.0]])

        result = lhs + rhs

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result._data, [11.0, 21.0, 31.0, 12.0, 22.0, 32.0])

    def test_broadcast_backward_reduces_over_expanded_dimensions(self):
        lhs = vg.tensor([[1.0], [2.0]], requires_grad=True)
        rhs = vg.tensor([[10.0, 20.0, 30.0]], requires_grad=True)

        (lhs + rhs).sum().backward()

        self.assertEqual(lhs.grad._data, [3.0, 3.0])
        self.assertEqual(rhs.grad._data, [2.0, 2.0, 2.0])

    def test_readme_cross_entropy_target_shape_is_accepted(self):
        logits = vg.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        # The README uses one-hot target tensors rather than a list of class indices.
        targets = vg.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        self.assertEqual(loss.shape, (1, 1))
        self.assertEqual(len(logits.grad._data), 4)


if __name__ == "__main__":
    unittest.main()
