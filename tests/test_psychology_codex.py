import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Assuming the module is located at lib/psychology_codex.py
# We import the target functions to be tested.
from lib.psychology_codex import map_interests_to_superpowers, evaluate_sdt_needs


@pytest.fixture
def sample_interests() -> List[str]:
    """
    Provides a standard list of kid-friendly interests for testing.
    These represent raw user inputs from the ArcMotivate app.
    """
    return ["drawing mythical creatures", "building lego robots", "helping stray animals"]


@pytest.fixture
def mock_superpower_response() -> Dict[str, Any]:
    """
    Provides a mocked response representing the Savickas career construction mapping.
    """
    return {
        "primary_superpower": "The Compassionate Creator",
        "core_themes": ["Artistic Expression", "Engineering", "Empathy"],
        "plot_twist_readiness": "High",
        "magical_items": ["Brush of Imagination", "Wrench of Wonder", "Cape of Caring"]
    }


class TestPsychologyCodex:
    """
    Test suite for the psychology_codex module.
    Validates the translation of Mark Savickas' career construction theory and 
    Self-Determination Theory (SDT) into gamified, kid-friendly concepts.
    """

    @patch("lib.psychology_codex._process_interest_data_flow")
    def test_map_interests_to_superpowers_success(
        self, 
        mock_process_data: MagicMock, 
        sample_interests: List[str], 
        mock_superpower_response: Dict[str, Any]
    ) -> None:
        """
        Tests that valid interests are successfully mapped to Superpowers.
        Patches the internal data flow to avoid mocking complex google-genai/adk internals.
        """
        # Arrange
        mock_process_data.return_value = mock_superpower_response

        # Act
        result: Dict[str, Any] = map_interests_to_superpowers(sample_interests)

        # Assert
        mock_process_data.assert_called_once_with(sample_interests)
        assert isinstance(result, dict), "Result must be a dictionary."
        assert result["primary_superpower"] == "The Compassionate Creator"
        assert len(result["magical_items"]) == 3
        assert "plot_twist_readiness" in result

    @patch("lib.psychology_codex._process_interest_data_flow")
    def test_map_interests_to_superpowers_empty_list(
        self, 
        mock_process_data: MagicMock
    ) -> None:
        """
        Tests the behavior of the mapping function when an empty list of interests is provided.
        The system should return a default 'Blank Canvas' superpower state.
        """
        # Arrange
        empty_response: Dict[str, Any] = {
            "primary_superpower": "The Blank Canvas",
            "core_themes": ["Discovery"],
            "plot_twist_readiness": "Pending",
            "magical_items": ["Magnifying Glass of Curiosity"]
        }
        mock_process_data.return_value = empty_response

        # Act
        result: Dict[str, Any] = map_interests_to_superpowers([])

        # Assert
        mock_process_data.assert_called_once_with([])
        assert result["primary_superpower"] == "The Blank Canvas"
        assert "Discovery" in result["core_themes"]

    def test_evaluate_sdt_needs_high_scores(self) -> None:
        """
        Tests SDT evaluation with high scores (e.g., 8-10).
        High scores should yield empowering, 'fully charged' feedback.
        """
        # Arrange
        autonomy: int = 9
        competence: int = 8
        relatedness: int = 10

        # Act
        result: Dict[str, str] = evaluate_sdt_needs(autonomy, competence, relatedness)

        # Assert
        assert isinstance(result, dict), "Result must be a dictionary."
        assert "autonomy_feedback" in result
        assert "competence_feedback" in result
        assert "relatedness_feedback" in result
        # Assuming the implementation returns specific keywords for high scores
        assert "Captain of your own ship" in result.get("autonomy_feedback", "") or result.get("autonomy_status") == "High"

    def test_evaluate_sdt_needs_low_scores(self) -> None:
        """
        Tests SDT evaluation with low scores (e.g., 1-3).
        Low scores should trigger gentle, growth-mindset 'Plot Twist' interventions.
        """
        # Arrange
        autonomy: int = 2
        competence: int = 3
        relatedness: int = 2

        # Act
        result: Dict[str, str] = evaluate_sdt_needs(autonomy, competence, relatedness)

        # Assert
        assert isinstance(result, dict)
        # Check that the feedback is encouraging rather than punitive
        assert "Plot Twist" in result.get("overall_assessment", "") or "recharge" in str(result).lower()

    @pytest.mark.parametrize("autonomy, competence, relatedness", [
        (-1, 5, 5),  # Negative score
        (5, 11, 5),  # Score above typical max (10)
        (0, 0, 0),   # Zero scores
    ])
    def test_evaluate_sdt_needs_boundary_values(
        self, 
        autonomy: int, 
        competence: int, 
        relatedness: int
    ) -> None:
        """
        Tests SDT evaluation with out-of-bounds or edge-case scores.
        The function should raise a ValueError for invalid psychological metrics.
        """
        # Act & Assert
        with pytest.raises(ValueError, match="Scores must be between 1 and 10"):
            evaluate_sdt_needs(autonomy, competence, relatedness)

    def test_evaluate_sdt_needs_type_enforcement(self) -> None:
        """
        Ensures that passing non-integer values to the SDT evaluator raises a TypeError.
        """
        # Act & Assert
        with pytest.raises(TypeError):
            # Ignoring type checking here intentionally to test runtime enforcement
            evaluate_sdt_needs("high", 5, 5) # type: ignore