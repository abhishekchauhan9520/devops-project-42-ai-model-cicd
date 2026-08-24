import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ml'))
from common import build_dataset, validate_dataset, FEATURES

class PipelineTests(unittest.TestCase):
    def test_dataset_contract(self):
        df = build_dataset(42)
        validate_dataset(df)
        self.assertEqual(list(df[FEATURES].columns), FEATURES)

    def test_training_is_reproducible(self):
        first = build_dataset(42)
        second = build_dataset(42)
        self.assertTrue(first.equals(second))

    def test_promotion_is_guarded(self):
        path = ROOT / 'ml' / 'promotion.py'
        self.assertIn('CONFIRM_PROMOTION', path.read_text())
        self.assertIn("choices=['candidate','champion']", path.read_text())

    def test_workflow_exists(self):
        workflow = ROOT / '.github' / 'workflows' / 'mlops.yml'
        self.assertTrue(workflow.exists())
        self.assertIn('quality-gate', workflow.read_text())

if __name__ == '__main__':
    unittest.main()
