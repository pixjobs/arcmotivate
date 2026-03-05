import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

# Assuming the module is available at lib.outcome_engine
from lib.outcome_engine import synthesize_blueprint, format_blueprint_for_ui


@pytest.fixture
def valid_journey_data() -> Dict[str, Any]:
    """
    Provides a standard set of journey data representing a user's progress
    through the ArcMotivate app, including plot twists and coaching insights.
    """
    return {
        "user_id": "hero_123",
        "story_theme": "Space Explorer",
        "coaching_insights": [
            "Shows great curiosity when exploring unknown planets.",
            "Collaborates well with alien species."
        ],
        "plot_twist_resolutions": [
            {
                "twist": "Ship engine failure",
                "resolution": "Used logic and patience to rewire the core.",
                "resilience_gained": 15
            }
        ]
    }


@pytest.fixture
def valid_superpowers() -> Dict[str, Any]:
    """
    Provides a standard set of discovered superpowers (skills/interests).
    """
    return {
        "primary": "Analytical Thinking",
        "secondary": "Empathy",
        "hidden_talent": "Creative Problem Solving"
    }


@pytest.fixture
def valid_blueprint_data() -> Dict[str, Any]:
    """
    Provides a synthesized blueprint dictionary ready for UI formatting.
    """
    return {
        "hero_title": "The Empathetic Space Engineer",
        "core_superpowers": ["Analytical Thinking", "Empathy", "Creative Problem Solving"],
        "resilience_level": "Titanium (Score: 90)",
        "career_paths": [
            "Aerospace Engineer",
            "Diplomatic Liaison",
            "Systems Architect"
        ],
        "action_steps": [
            "Join a local robotics club.",
            "Read a book about emotional intelligence."
        ]
    }


class TestSynthesizeBlueprint:
    """
    Test suite for the synthesize_blueprint function in the Outcome Engine.
    Ensures data from the user's journey is correctly merged into a cohesive blueprint.
    """

    @patch("lib.outcome_engine._generate_career_insights")
    def test_synthesize_blueprint_success(
        self,
        mock_generate_insights: MagicMock,
        valid_journey_data: Dict[str, Any],
        valid_superpowers: Dict[str, Any]
    ) -> None:
        """
        Verifies that synthesize_blueprint correctly aggregates inputs and 
        returns a properly structured blueprint dictionary.
        """
        # Mocking internal data flow to avoid complex LLM/GenAI calls
        mock_generate_insights.return_value = ["Aerospace Engineer", "Diplomatic Liaison"]
        
        resilience_score = 90
        
        result = synthesize_blueprint(
            journey_data=valid_journey_data,
            superpowers=valid_superpowers,
            resilience_score=resilience_score
        )
        
        assert isinstance(result, dict), "Blueprint must be a dictionary."
        assert "hero_title" in result, "Blueprint is missing 'hero_title'."
        assert "career_paths" in result, "Blueprint is missing 'career_paths'."
        assert result["resilience_score"] == 90, "Resilience score was not mapped correctly."
        mock_generate_insights.assert_called_once()

    def test_synthesize_blueprint_empty_journey(self, valid_superpowers: Dict[str, Any]) -> None:
        """
        Verifies behavior when journey_data is empty. The engine should still
        generate a baseline blueprint using superpowers and resilience.
        """
        empty_journey: Dict[str, Any] = {}
        resilience_score = 50
        
        result = synthesize_blueprint(
            journey_data=empty_journey,
            superpowers=valid_superpowers,
            resilience_score=resilience_score
        )
        
        assert isinstance(result, dict)
        assert result.get("resilience_score") == 50
        assert "core_superpowers" in result

    def test_synthesize_blueprint_zero_resilience(
        self, 
        valid_journey_data: Dict[str, Any], 
        valid_superpowers: Dict[str, Any]
    ) -> None:
        """
        Verifies edge case where resilience score is 0.
        """
        result = synthesize_blueprint(
            journey_data=valid_journey_data,
            superpowers=valid_superpowers,
            resilience_score=0
        )
        
        assert result["resilience_score"] == 0
        # Ensure the system provides an encouraging baseline status rather than failing
        assert "resilience_status" in result


class TestFormatBlueprintForUI:
    """
    Test suite for the format_blueprint_for_ui function.
    Ensures the dictionary is transformed into a visually engaging, kid-friendly string.
    """

    def test_format_blueprint_for_ui_success(self, valid_blueprint_data: Dict[str, Any]) -> None:
        """
        Verifies that a valid blueprint dictionary is formatted into a rich string
        containing expected headers and emojis.
        """
        result = format_blueprint_for_ui(blueprint_data=valid_blueprint_data)
        
        assert isinstance(result, str), "Formatted output must be a string."
        assert "The Empathetic Space Engineer" in result, "Hero title missing from UI output."
        assert "Titanium (Score: 90)" in result, "Resilience level missing from UI output."
        assert "Aerospace Engineer" in result, "Career paths missing from UI output."
        
        # Check for gamified/kid-friendly elements (assuming the function adds these)
        assert "🌟" in result or "🚀" in result or "✨" in result, "UI output should contain engaging emojis."

    def test_format_blueprint_for_ui_missing_fields(self) -> None:
        """
        Verifies that the formatter handles missing fields gracefully without raising KeyErrors,
        providing fallback text instead.
        """
        incomplete_blueprint = {
            "hero_title": "Mystery Explorer"
            # Missing career_paths, core_superpowers, etc.
        }
        
        result = format_blueprint_for_ui(blueprint_data=incomplete_blueprint)
        
        assert isinstance(result, str)
        assert "Mystery Explorer" in result
        assert "Paths are still shrouded in mystery" in result or "Keep exploring" in result, \
            "Formatter should provide fallback text for missing career paths."

    def test_format_blueprint_for_ui_empty_dict(self) -> None:
        """
        Verifies that an entirely empty dictionary returns a safe, default placeholder string.
        """
        result = format_blueprint_for_ui(blueprint_data={})
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Awaiting Discovery" in result or "Blueprint Empty" in result, \
            "Formatter should handle completely empty data safely."