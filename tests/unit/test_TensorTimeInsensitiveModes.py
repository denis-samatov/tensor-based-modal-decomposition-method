import unittest

import torch

from TBMD.core.modal_processor.modes import (
    BatchModalProcessor,
    ModalProcessorConfig,
    ModalTensorStacker,
    ProcessingStrategy,
    TimeInsensitiveModeComputer,
)


class TestTensorTimeInsensitiveModes(unittest.TestCase):
    def setUp(self):
        self.cores = {
            "subject1": torch.randn(5, 5, 3, 5),
            "subject2": torch.randn(5, 5, 3, 5),
        }
        self.factors = {
            "subject1": [
                torch.randn(10, 5),
                torch.randn(10, 5),
                torch.randn(5, 3),
                torch.randn(10, 5),
            ],
            "subject2": [
                torch.randn(10, 5),
                torch.randn(10, 5),
                torch.randn(5, 3),
                torch.randn(10, 5),
            ],
        }
        self.config = ModalProcessorConfig(
            processing_strategy=ProcessingStrategy.BATCH,
        )

    def test_BatchModalProcessor(self):
        processor = BatchModalProcessor(self.config)
        modal_tensors = processor.process_multiple_subjects(self.cores, self.factors)
        self.assertIn("subject1", modal_tensors)
        self.assertEqual(modal_tensors["subject1"].shape, (10, 10, 5, 5))

    def test_ModalTensorStacker(self):
        processor = BatchModalProcessor(self.config)
        modal_tensors = processor.process_multiple_subjects(self.cores, self.factors)
        stacker = ModalTensorStacker(self.config)
        A_tensor = stacker.stack_modal_tensors(modal_tensors)
        self.assertEqual(A_tensor.shape, (10, 10, 5, 10))

    def test_fourth_order_mode_matches_explicit_three_factor_contraction(self):
        """The 4D path must retain both spatial modes and the property mode."""
        factors = [
            torch.randn(4, 2),
            torch.randn(5, 3),
            torch.randn(2, 2),
        ]
        core_slice = torch.randn(2, 3, 2)

        actual = TimeInsensitiveModeComputer(self.config).compute_single_mode(
            factors, core_slice
        )
        expected = torch.einsum(
            "ia,jb,kc,abc->ijk", factors[0], factors[1], factors[2], core_slice
        )

        self.assertEqual(actual.shape, (4, 5, 2))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
