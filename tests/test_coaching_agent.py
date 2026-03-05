"""
ArcMotivate - Elite QA Test Suite for Coaching Agent
====================================================

This test suite validates the `lib.coaching_agent` module, ensuring our Socratic mentor 
behaves magically and reliably for our young users (ages 8-18). 

QA Strategy:
- Isolate the LLM: We use `unittest.mock.MagicMock` to patch the `google.genai.Client` 
  data flows, avoiding brittle dependencies on the complex internals of `google-genai`.
- Edge Cases: We test empty contexts, malformed LLM JSON responses, and API failures 
  to ensure the app gracefully handles unexpected 'Plot Twists'.
- Type Safety: Fully type-hinted test cases for maintainability.
"""

import json
import pytest
from typing import Any, Generator
from unittest.mock import MagicMock, patch

# Assuming the module is structured to import genai and use genai.Client()
# If the implementation uses a different internal wrapper, the patch target can be adjusted.
from lib.coaching_agent import generate_socratic_question, analyze_response


@pytest.fixture
def mock_genai_client() -> Generator[MagicMock, None, None]:
    """
    Patches the google-genai Client to intercept internal data flows.
    Yields the mocked client instance so individual tests can configure its behavior.
    """
    with patch("lib.coaching_agent.genai.Client") as mock_client_class:
        mock_instance = mock_client_class.return_value
        
        # Setup a default successful response structure
        mock_response = MagicMock()
        mock_response.text = "What kind of magical creature would you want as a co-worker?"
        mock_instance.models.generate_content.return_value = mock_response
        
        yield mock_instance


class TestGenerateSocraticQuestion:
    """Test suite for the generate_socratic_question function."""

    def test_generate_socratic_question_success(self, mock_genai_client: MagicMock) -> None:
        """
        Validates that a rich user context successfully generates an imaginative, 
        open-ended Socratic question.
        """
        user_context: dict[str, Any] = {
            "age": 12,
            "interests": ["space exploration", "drawing dragons"],
            "current_quest": "Discover Your Superpower"
        }
        
        # Configure the mock for this specific test
        expected_question = "If you could draw a dragon that survives in outer space, what would its wings look like?"
        mock_genai_client.models.generate_content.return_value.text = expected_question
        
        result = generate_socratic_question(user_context)
        
        assert isinstance(result, str)
        assert result == expected_question
        
        # Verify the model was called with the correct model ID
        mock_genai_client.models.generate_content.assert_called_once()
        call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
        assert call_kwargs.get("model") == "gemini-3.1-flash-lite-preview"

    def test_generate_socratic_question_empty_context(self, mock_genai_client: MagicMock) -> None:
        """
        Validates that the agent can still generate a baseline exploratory question 
        even if the user context is completely empty (e.g., a brand new user).
        """
        user_context: dict[str, Any] = {}
        
        expected_question = "If you had a magic wand that could build anything, what is the first thing you'd create?"
        mock_genai_client.models.generate_content.return_value.text = expected_question
        
        result = generate_socratic_question(user_context)
        
        assert isinstance(result, str)
        assert result == expected_question
        mock_genai_client.models.generate_content.assert_called_once()

    def test_generate_socratic_question_api_failure(self, mock_genai_client: MagicMock) -> None:
        """
        Validates resilience. If the LLM API throws an exception (a sudden Plot Twist!), 
        the function should catch it and return a safe, default fallback question.
        """
        # Simulate an API timeout or failure
        mock_genai_client.models.generate_content.side_effect = Exception("API Timeout")
        
        user_context: dict[str, Any] = {"age": 14}
        result = generate_socratic_question(user_context)
        
        # The system should fallback gracefully rather than crashing the app
        assert isinstance(result, str)
        assert len(result) > 0
        assert "What is something you love doing" in result or "?" in result


class TestAnalyzeResponse:
    """Test suite for the analyze_response function."""

    def test_analyze_response_success(self, mock_genai_client: MagicMock) -> None:
        """
        Validates that a thoughtful user response is correctly analyzed and parsed 
        into a structured dictionary of interests and traits.
        """
        user_input = "I really like building forts out of pillows and figuring out how to make them not fall down."
        
        # The LLM is expected to return a JSON string for analysis
        mock_json_response = {
            "discovered_interests": ["architecture", "engineering", "problem-solving"],
            "sentiment": "enthusiastic",
            "suggested_plot_twist": "A sudden windstorm tests the fort's structural integrity!"
        }
        mock_genai_client.models.generate_content.return_value.text = json.dumps(mock_json_response)
        
        result = analyze_response(user_input)
        
        assert isinstance(result, dict)
        assert "discovered_interests" in result
        assert "engineering" in result["discovered_interests"]
        assert result["sentiment"] == "enthusiastic"

    def test_analyze_response_short_input(self, mock_genai_client: MagicMock) -> None:
        """
        Validates behavior when the user gives a very short or unhelpful response.
        The agent should recognize the lack of data.
        """
        user_input = "idk"
        
        mock_json_response = {
            "discovered_interests": [],
            "sentiment": "neutral",
            "needs_follow_up": True
        }
        mock_genai_client.models.generate_content.return_value.text = json.dumps(mock_json_response)
        
        result = analyze_response(user_input)
        
        assert isinstance(result, dict)
        assert len(result.get("discovered_interests", [])) == 0
        assert result.get("needs_follow_up") is True

    def test_analyze_response_malformed_json(self, mock_genai_client: MagicMock) -> None:
        """
        Validates resilience against LLM hallucinations. If the model returns 
        malformed JSON, the function should handle the JSONDecodeError and return 
        a safe default dictionary.
        """
        user_input = "I want to be a space pirate!"
        
        # Simulate the LLM returning plain text instead of the requested JSON schema
        mock_genai_client.models.generate_content.return_value.text = "That sounds like a great career! You like space and pirates."
        
        result = analyze_response(user_input)
        
        # Should fallback to a safe default dict rather than raising an unhandled exception
        assert isinstance(result, dict)
        assert "error" in result or "discovered_interests" in result
        # If it defaults to empty interests on failure:
        if "discovered_interests" in result:
            assert isinstance(result["discovered_interests"], list)

    def test_analyze_response_empty_input(self, mock_genai_client: MagicMock) -> None:
        """
        Validates that an empty string input is handled properly, potentially 
        bypassing the LLM call entirely to save tokens.
        """
        user_input = "   "
        
        result = analyze_response(user_input)
        
        assert isinstance(result, dict)
        # Depending on implementation, it might not even call the LLM for empty strings
        # If it does, we just ensure it returns a valid dict.
        assert isinstance(result.get("discovered_interests", []), list)