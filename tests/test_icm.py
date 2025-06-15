"""
Test the ICM core functionality.
"""

import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch

from icm.datasets import ICMDataset, ICMExample, create_synthetic_dataset
from icm.consistency import LogicalConsistencyChecker, MathConsistencyRule
from icm.storage import ICMStorage
from icm.core import ICMResult


class TestICMDatasets(unittest.TestCase):
    """Test ICM dataset functionality."""
    
    def test_icm_example_creation(self):
        """Test ICMExample creation."""
        example = ICMExample("What is 2+2?", {"category": "math"})
        self.assertEqual(example.input_text, "What is 2+2?")
        self.assertEqual(example.metadata["category"], "math")
    
    def test_icm_dataset_creation(self):
        """Test ICMDataset creation and methods."""
        examples = [
            ICMExample("Question 1", {"id": 1}),
            ICMExample("Question 2", {"id": 2}),
            ICMExample("Question 3", {"id": 3})
        ]
        dataset = ICMDataset(examples)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0].input_text, "Question 1")
        
        # Test sampling
        sampled = dataset.sample(2, seed=42)
        self.assertEqual(len(sampled), 2)
        
        # Test filtering
        filtered = dataset.filter_by_metadata("id", 1)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].metadata["id"], 1)
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset("math", num_examples=10, seed=42)
        self.assertEqual(len(dataset), 20)  # 10 correct + 10 incorrect
        
        # Check that we have both True and False labels in metadata
        labels = [ex.metadata["gold_label"] for ex in dataset.examples]
        self.assertIn("True", labels)
        self.assertIn("False", labels)


class TestConsistencyChecker(unittest.TestCase):
    """Test logical consistency checking."""
    
    def test_math_consistency_rule(self):
        """Test mathematical consistency rule."""
        rule = MathConsistencyRule()
        
        # Create examples with same question but different answers
        example1 = ICMExample("Question: What is 2+2?\nClaim: 2+2 = 4", {})
        example2 = ICMExample("Question: What is 2+2?\nClaim: 2+2 = 5", {})
        
        # Both can't be True if answers are different
        self.assertTrue(rule.check(example1, example2, "True", "False"))
        self.assertTrue(rule.check(example1, example2, "False", "True"))
        self.assertFalse(rule.check(example1, example2, "True", "True"))
    
    def test_consistency_checker(self):
        """Test overall consistency checker."""
        checker = LogicalConsistencyChecker()
        
        # Create test examples
        example1 = ICMExample("Question: What is 2+2?\nClaim: 2+2 = 4", {})
        example2 = ICMExample("Question: What is 2+2?\nClaim: 2+2 = 5", {})
        
        # Test consistency checking
        self.assertTrue(checker.check_consistency(example1, example2, "True", "False"))
        self.assertFalse(checker.check_consistency(example1, example2, "True", "True"))
        
        # Test getting consistent options
        options = checker.get_consistent_options(example1, example2, "True", "True")
        self.assertIn(("True", "False"), options)
        self.assertIn(("False", "True"), options)


class TestICMStorage(unittest.TestCase):
    """Test ICM storage functionality."""
    
    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ICMStorage(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_result(self):
        """Test saving and loading ICM results."""
        # Create test result
        labeled_examples = [
            {"input": "Question 1", "label": "True", "metadata": {"id": 1}},
            {"input": "Question 2", "label": "False", "metadata": {"id": 2}}
        ]
        
        result = ICMResult(
            labeled_examples=labeled_examples,
            score=75.5,
            iterations=100,
            convergence_info={"final_temp": 0.01},
            metadata={"model": "test-model"}
        )
        
        # Save result
        file_path = self.storage.save_result(result, "test_result")
        self.assertTrue(os.path.exists(file_path))
        
        # Load result
        loaded_result = self.storage.load_result(file_path)
        
        self.assertEqual(len(loaded_result.labeled_examples), 2)
        self.assertEqual(loaded_result.score, 75.5)
        self.assertEqual(loaded_result.iterations, 100)
        self.assertEqual(loaded_result.metadata["model"], "test-model")
    
    def test_save_labeled_dataset(self):
        """Test saving labeled dataset in different formats."""
        labeled_examples = [
            {"input": "Question 1", "label": "True", "metadata": {"id": 1}},
            {"input": "Question 2", "label": "False", "metadata": {"id": 2}}
        ]
        
        # Test JSONL format
        jsonl_path = self.storage.save_labeled_dataset(labeled_examples, "test_dataset", "jsonl")
        self.assertTrue(os.path.exists(jsonl_path))
        
        # Test JSON format
        json_path = self.storage.save_labeled_dataset(labeled_examples, "test_dataset", "json")
        self.assertTrue(os.path.exists(json_path))


class TestICMSearcher(unittest.TestCase):
    """Test ICM searcher functionality."""
    
    @patch('icm.core.AutoModelForCausalLM')
    @patch('icm.core.AutoTokenizer')
    def test_icm_searcher_initialization(self, mock_tokenizer, mock_model):
        """Test ICM searcher initialization."""
        # Mock the tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        from icm.core import ICMSearcher
        
        searcher = ICMSearcher(
            model_name="test-model",
            alpha=30.0,
            max_iterations=10
        )
        
        self.assertEqual(searcher.alpha, 30.0)
        self.assertEqual(searcher.max_iterations, 10)
        self.assertIsNotNone(searcher.consistency_checker)


if __name__ == "__main__":
    unittest.main()
