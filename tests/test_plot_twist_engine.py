"""
ArcMotivate - Plot Twist Engine Test Suite
==========================================

This test suite validates the behavior of `lib/plot_twist_engine.py`.
It ensures that 'Curve Balls' are correctly generated based on the user's
career path and level, and that resilience responses are accurately assessed.

Internal data flows are patched using `unittest.mock.MagicMock` to avoid
complex dependencies on `google-genai` or `google-adk`.
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the functions to be tested from the plot twist engine.
from lib.plot_twist_engine import trigger_curve_ball, assess_resilience_response


@pytest.fixture
def sample_curve_ball() -> Dict[str, str]:
    """
    Provides a standard curve ball response dictionary for testing.
    """
    return {
        "scenario_id": "cb_tech_001",
        "title": "Sudden Tech Shift!",
        "description": "The programming language you spent months learning was just deprecated. What is your next move?",
        "category": "Adaptability"
    }


@pytest.fixture
def sample_excellent_assessment() -> Dict[str, Any]:
    """
    Provides a standard assessment response for a highly resilient user solution.
    """
    return {
        "scenario_id": "cb_tech_001",
        "resilience_score": 95,
        "feedback": "Incredible adaptability! Pivoting to a new language while applying your core logic skills is a pro move.",
        "xp_awarded": 150,
        "plot_twist_survived": True
    }


@pytest.fixture
def sample_poor_assessment() -> Dict[str, Any]:
    """
    Provides a standard assessment response for a low-resilience user solution.
    """
    return {
        "scenario_id": "cb_tech_001",
        "resilience_score": 30,
        "feedback": "It's normal to feel frustrated, but giving up won't help you level up. How about looking for similarities in a new language?",
        "xp_awarded": 20,
        "plot_twist_survived": False
    }


class TestTriggerCurveBall:
    """
    Test cases for the `trigger_curve_ball` function.
    """

    @patch("lib.plot_twist_engine._fetch_scenario_data")
    def test_trigger_curve_ball_standard_path(
        self, mock_fetch_scenario: MagicMock, sample_curve_ball: Dict[str, str]
    ) -> None:
        """
        Test that a curve ball is successfully generated for a valid path and level.
        """
        # Arrange
        mock_fetch_scenario.return_value = sample_curve_ball
        current_path = "Software Engineer"
        user_level = 5

        # Act
        result = trigger_curve_ball(current_path, user_level)

        # Assert
        mock_fetch_scenario.assert_called_once_with(current_path, user_level)
        assert isinstance(result, dict)
        assert result["scenario_id"] == "cb_tech_001"
        assert result["title"] == "Sudden Tech Shift!"
        assert "deprecated" in result["description"]

    @patch("lib.plot_twist_engine._fetch_scenario_data")
    def test_trigger_curve_ball_high_level_scaling(
        self, mock_fetch_scenario: MagicMock
    ) -> None:
        """
        Test that the engine handles high-level users appropriately,
        expecting a more complex scenario data flow.
        """
        # Arrange
        mock_fetch_scenario.return_value = {
            "scenario_id": "cb_lead_099",
            "title": "Team Mutiny",
            "description": "Two senior developers quit right before launch. Handle it, Lead!",
            "category": "Leadership Crisis"
        }
        current_path = "Game Director"
        user_level = 50

        # Act
        result = trigger_curve_ball(current_path, user_level)

        # Assert
        mock_fetch_scenario.assert_called_once_with(current_path, user_level)
        assert result["category"] == "Leadership Crisis"
        assert result["scenario_id"] == "cb_lead_099"

    def test_trigger_curve_ball_invalid_level(self) -> None:
        """
        Test that an invalid user level raises a ValueError.
        """
        # Arrange
        current_path = "Marine Biologist"
        invalid_level = -3

        # Act & Assert
        with pytest.raises(ValueError, match="User level must be a positive integer."):
            trigger_curve_ball(current_path, invalid_level)

    def test_trigger_curve_ball_empty_path(self) -> None:
        """
        Test that an empty career path raises a ValueError.
        """
        # Arrange
        current_path = ""
        user_level = 10

        # Act & Assert
        with pytest.raises(ValueError, match="Current path cannot be empty."):
            trigger_curve_ball(current_path, user_level)


class TestAssessResilienceResponse:
    """
    Test cases for the `assess_resilience_response` function.
    """

    @patch("lib.plot_twist_engine._evaluate_solution_internals")
    def test_assess_resilience_response_excellent(
        self, mock_evaluate: MagicMock, sample_excellent_assessment: Dict[str, Any]
    ) -> None:
        """
        Test assessment of a highly adaptable and resilient user solution.
        """
        # Arrange
        mock_evaluate.return_value = sample_excellent_assessment
        scenario_id = "cb_tech_001"
        user_solution = "I would read the documentation for the new standard language and map my existing knowledge to the new syntax."

        # Act
        result = assess_resilience_response(scenario_id, user_solution)

        # Assert
        mock_evaluate.assert_called_once_with(scenario_id, user_solution)
        assert result["resilience_score"] == 95
        assert result["plot_twist_survived"] is True
        assert result["xp_awarded"] > 100

    @patch("lib.plot_twist_engine._evaluate_solution_internals")
    def test_assess_resilience_response_needs_work(
        self, mock_evaluate: MagicMock, sample_poor_assessment: Dict[str, Any]
    ) -> None:
        """
        Test assessment of a poor solution, ensuring constructive feedback is provided.
        """
        # Arrange
        mock_evaluate.return_value = sample_poor_assessment
        scenario_id = "cb_tech_001"
        user_solution = "I would just quit and play video games instead."

        # Act
        result = assess_resilience_response(scenario_id, user_solution)

        # Assert
        mock_evaluate.assert_called_once_with(scenario_id, user_solution)
        assert result["resilience_score"] == 30
        assert result["plot_twist_survived"] is False
        assert "frustrated" in result["feedback"]

    def test_assess_resilience_response_empty_solution(self) -> None:
        """
        Test that an empty user solution raises a ValueError.
        """
        # Arrange
        scenario_id = "cb_tech_001"
        empty_solution = "   "

        # Act & Assert
        with pytest.raises(ValueError, match="User solution cannot be empty."):
            assess_resilience_response(scenario_id, empty_solution)

    def test_assess_resilience_response_invalid_scenario(self) -> None:
        """
        Test that an empty scenario ID raises a ValueError.
        """
        # Arrange
        scenario_id = ""
        user_solution = "I will try my best!"

        # Act & Assert
        with pytest.raises(ValueError, match="Scenario ID cannot be empty."):
            assess_resilience_response(scenario_id, user_solution)