"""
Logical Consistency Checker for ICM.

This module implements logical consistency checks to prevent degenerate solutions
in the ICM algorithm by enforcing simple logical constraints.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod

from .datasets import ICMExample


class ConsistencyRule(ABC):
    """Abstract base class for consistency rules."""
    
    @abstractmethod
    def check(self, example1: ICMExample, example2: ICMExample, label1: str, label2: str) -> bool:
        """Check if two labeled examples are consistent according to this rule."""
        pass
    
    @abstractmethod
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """Get all consistent label combinations for two examples."""
        pass
    
    @abstractmethod
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if this rule applies to the relationship between two examples."""
        pass


class MathConsistencyRule(ConsistencyRule):
    """Consistency rule for mathematical problems."""
    
    def check(self, example1: ICMExample, example2: ICMExample, label1: str, label2: str) -> bool:
        """Check mathematical consistency."""
        # If both examples are solutions to the same problem
        if self._same_math_problem(example1, example2):
            # Extract final answers
            answer1 = self._extract_final_answer(example1.input_text)
            answer2 = self._extract_final_answer(example2.input_text)
            
            if answer1 and answer2 and answer1 != answer2:
                # Different final answers cannot both be True
                return not (label1 == "True" and label2 == "True")
        
        return True  # No constraint violated
    
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """Get consistent label options for math problems."""
        if self._same_math_problem(example1, example2):
            answer1 = self._extract_final_answer(example1.input_text)
            answer2 = self._extract_final_answer(example2.input_text)
            
            if answer1 and answer2 and answer1 != answer2:
                # Different answers: at most one can be True
                return [("True", "False"), ("False", "True"), ("False", "False")]
        
        # Default: all combinations allowed
        return [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
    
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if examples are related math problems."""
        return self._same_math_problem(example1, example2)
    
    def _same_math_problem(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if two examples are solutions to the same math problem."""
        # Look for common problem statement
        lines1 = example1.input_text.split('\n')
        lines2 = example2.input_text.split('\n')
        
        # Find question lines (usually start with keywords or end with ?)
        question1 = self._extract_question(lines1)
        question2 = self._extract_question(lines2)
        
        if question1 and question2:
            # Simple similarity check (could be improved)
            return question1.strip() == question2.strip()
        
        return False
    
    def _extract_question(self, lines: List[str]) -> Optional[str]:
        """Extract the question from text lines."""
        for line in lines:
            line = line.strip()
            if line.endswith('?') or 'Question:' in line:
                return line.replace('Question:', '').strip()
        return None
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Extract the final numerical answer from text."""
        # Look for mathematical equations in the claim
        equation_patterns = [
            r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)",  # Addition
            r"(\d+)\s*-\s*(\d+)\s*=\s*(\d+)",  # Subtraction  
            r"(\d+)\s*\*\s*(\d+)\s*=\s*(\d+)",  # Multiplication
            r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+)",  # Division
        ]
        
        for pattern in equation_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(3)  # Return the claimed result
        
        # Fallback to original patterns
        answer_patterns = [
            r"The answer is ([0-9]+(?:\.[0-9]+)?)",
            r"Therefore,?\s*([0-9]+(?:\.[0-9]+)?)",
            r"= ([0-9]+(?:\.[0-9]+)?)\s*$",
            r"answer:\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


class ComparisonConsistencyRule(ConsistencyRule):
    """Consistency rule for comparison tasks (asymmetry)."""
    
    def check(self, example1: ICMExample, example2: ICMExample, label1: str, label2: str) -> bool:
        """Check comparison consistency (asymmetry)."""
        if self._are_inverse_comparisons(example1, example2):
            # A > B and B > A cannot both be True
            return not (label1 == "True" and label2 == "True")
        return True
    
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """Get consistent options for comparison tasks."""
        if self._are_inverse_comparisons(example1, example2):
            # Asymmetry: at most one can be True
            return [("True", "False"), ("False", "True"), ("False", "False")]
        
        return [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
    
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if examples are inverse comparisons."""
        return self._are_inverse_comparisons(example1, example2)
    
    def _are_inverse_comparisons(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Check if two examples are inverse comparisons (A > B vs B > A)."""
        # Extract entities being compared
        entities1 = self._extract_comparison_entities(example1.input_text)
        entities2 = self._extract_comparison_entities(example2.input_text)
        
        if entities1 and entities2 and len(entities1) == 2 and len(entities2) == 2:
            # Check if they're inverse: (A, B) vs (B, A)
            return (entities1[0] == entities2[1] and entities1[1] == entities2[0])
        
        return False
    
    def _extract_comparison_entities(self, text: str) -> Optional[List[str]]:
        """Extract entities being compared from text."""
        # Look for patterns like "Response A" and "Response B"
        pattern = r"Response ([AB])"
        matches = re.findall(pattern, text)
        
        if len(matches) == 2:
            return [f"Response {matches[0]}", f"Response {matches[1]}"]
        
        # Look for other comparison patterns
        # This is a simplified version - could be enhanced
        if " vs " in text.lower():
            parts = text.lower().split(" vs ")
            if len(parts) == 2:
                return [parts[0].strip(), parts[1].strip()]
        
        return None


class GenericConsistencyRule(ConsistencyRule):
    """Generic consistency rule - allows all combinations."""
    
    def check(self, example1: ICMExample, example2: ICMExample, label1: str, label2: str) -> bool:
        """Generic rule - always consistent."""
        return True
    
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """All combinations are allowed."""
        return [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
    
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """Generic rule applies to all pairs."""
        return True


class LogicalConsistencyChecker:
    """
    Main consistency checker that applies multiple consistency rules.
    """
    
    def __init__(self, rules: Optional[List[ConsistencyRule]] = None):
        """
        Initialize with consistency rules.
        
        Args:
            rules: List of consistency rules to apply. If None, uses default rules.
        """
        if rules is None:
            self.rules = [
                MathConsistencyRule(),
                ComparisonConsistencyRule(),
                GenericConsistencyRule()  # Fallback rule
            ]
        else:
            self.rules = rules
    
    def check_consistency(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        label1: str, 
        label2: str
    ) -> bool:
        """
        Check if two labeled examples are consistent.
        
        Args:
            example1: First example
            example2: Second example  
            label1: Label for first example
            label2: Label for second example
            
        Returns:
            True if consistent, False otherwise
        """
        for rule in self.rules:
            if rule.has_relationship(example1, example2):
                if not rule.check(example1, example2, label1, label2):
                    return False
        return True
    
    def get_consistent_options(
        self, 
        example1: ICMExample, 
        example2: ICMExample, 
        current_label1: str, 
        current_label2: str
    ) -> List[Tuple[str, str]]:
        """
        Get all consistent label combinations for two examples.
        
        Args:
            example1: First example
            example2: Second example
            current_label1: Current label for first example
            current_label2: Current label for second example
            
        Returns:
            List of (label1, label2) tuples that are consistent
        """
        # Start with all possible combinations
        all_options = [("True", "True"), ("True", "False"), ("False", "True"), ("False", "False")]
        
        # Apply each rule to filter options
        for rule in self.rules:
            if rule.has_relationship(example1, example2):
                rule_options = rule.get_consistent_options(example1, example2, current_label1, current_label2)
                # Intersect with current options
                all_options = [opt for opt in all_options if opt in rule_options]
        
        return all_options if all_options else [("False", "False")]  # Fallback
    
    def has_relationship(self, example1: ICMExample, example2: ICMExample) -> bool:
        """
        Check if any rule establishes a relationship between two examples.
        
        Args:
            example1: First example
            example2: Second example
            
        Returns:
            True if there's a relationship, False otherwise
        """
        for rule in self.rules:
            if rule.has_relationship(example1, example2):
                return True
        return False
    
    def add_rule(self, rule: ConsistencyRule):
        """Add a new consistency rule."""
        self.rules.insert(-1, rule)  # Insert before the generic rule
    
    def count_inconsistencies(self, labeled_examples: List[Dict[str, Any]]) -> int:
        """Count total inconsistencies in a set of labeled examples."""
        count = 0
        n = len(labeled_examples)
        
        for i in range(n):
            for j in range(i + 1, n):
                example1 = ICMExample(labeled_examples[i]["input"], labeled_examples[i].get("metadata", {}))
                example2 = ICMExample(labeled_examples[j]["input"], labeled_examples[j].get("metadata", {}))
                label1 = labeled_examples[i]["label"]
                label2 = labeled_examples[j]["label"]
                
                if not self.check_consistency(example1, example2, label1, label2):
                    count += 1
        
        return count
