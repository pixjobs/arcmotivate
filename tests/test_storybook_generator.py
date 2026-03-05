"""
Test Suite for ArcMotivate Storybook Generator (lib/storybook_generator.py)

This suite verifies the generation of personalized 'Hero's Journey' narratives
and neon pixel-art illustrations for kids and teens. It uses MagicMock to patch
internal data flows to the Gemini API, ensuring tests are fast, deterministic,
and avoid mocking the complex, deeply nested internals of the google-genai SDK.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

# Target imports
from lib.storybook_generator import (
    generate_heros_journey_text,
    generate_pixel_art_illustration,
)


@pytest.fixture
def valid_user_profile() -> Dict[str, Any]:
    """
    Provides a standard, valid user profile for a teen user in ArcMotivate.
    Includes interests and 'Plot Twists' (resilience-building events).
    """
    return {
        "user_id": "usr_arc_8891",
        "name": "Alex",
        "age": 14,
        "interests": ["robotics", "space exploration", "retro video games"],
        "dream_career": "Intergalactic Robotics Engineer",
        "plot_twists_overcome": [
            "Failed the middle school science fair",
            "Overcame fear of public speaking during a coding demo"
        ]
    }


@pytest.fixture
def empty_user_profile() -> Dict[str, Any]:
    """
    Provides an empty user profile to test edge cases and fallback behaviors.
    """
    return {}


@patch("lib.storybook_generator.genai.Client")
def test_generate_heros_journey_text_success(
    mock_genai_client: MagicMock,
    valid_user_profile: Dict[str, Any]
) -> None:
    """
    Tests successful generation of a Hero's Journey narrative.
    Verifies that the correct model (gemini-3.1-flash-lite-preview) is used
    and that user profile data is injected into the prompt.
    """
    # Arrange
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    expected_story = "In the neon-lit corridors of the future, Alex built a robot..."
    
    # Patching the internal data flow (the text attribute of the response)
    mock_response.text = expected_story
    mock_client_instance.models.generate_content.return_value = mock_response

    # Act
    result = generate_heros_journey_text(valid_user_profile)

    # Assert
    assert result == expected_story
    mock_client_instance.models.generate_content.assert_called_once()
    
    call_kwargs = mock_client_instance.models.generate_content.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-3.1-flash-lite-preview"
    
    # Verify prompt construction includes key user details
    prompt_contents = str(call_kwargs.get("contents", ""))
    assert "Alex" in prompt_contents
    assert "Intergalactic Robotics Engineer" in prompt_contents
    assert "Failed the middle school science fair" in prompt_contents


@patch("lib.storybook_generator.genai.Client")
def test_generate_heros_journey_text_empty_profile(
    mock_genai_client: MagicMock,
    empty_user_profile: Dict[str, Any]
) -> None:
    """
    Tests narrative generation when the user profile is missing data.
    The generator should handle this gracefully using default generic prompts.
    """
    # Arrange
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    expected_story = "A mysterious hero embarks on a grand adventure..."
    mock_response.text = expected_story
    mock_client_instance.models.generate_content.return_value = mock_response

    # Act
    result = generate_heros_journey_text(empty_user_profile)

    # Assert
    assert result == expected_story
    mock_client_instance.models.generate_content.assert_called_once()


@patch("lib.storybook_generator.genai.Client")
def test_generate_heros_journey_text_api_failure(
    mock_genai_client: MagicMock,
    valid_user_profile: Dict[str, Any]
) -> None:
    """
    Tests that the function surfaces or handles exceptions when the LLM API fails.
    """
    # Arrange
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_content.side_effect = Exception("API Timeout")

    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        generate_heros_journey_text(valid_user_profile)
    
    assert "API Timeout" in str(exc_info.value)


@patch("lib.storybook_generator.genai.Client")
def test_generate_pixel_art_illustration_success(
    mock_genai_client: MagicMock
) -> None:
    """
    Tests successful generation of a pixel-art illustration.
    Verifies that the correct image model (gemini-3.1-flash-image-preview) is used
    and that the Geocities/Techy aesthetic is enforced in the prompt.
    """
    # Arrange
    scene_description = "A neon-lit robotics lab in space."
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    
    # Patching the internal data flow for image generation
    mock_image = MagicMock()
    # Supporting standard SDK return structures (bytes or b64 string)
    mock_image.image.image_bytes = b"fake_neon_pixel_art_bytes"
    mock_response.generated_images = [mock_image]
    mock_client_instance.models.generate_images.return_value = mock_response

    # Act
    result = generate_pixel_art_illustration(scene_description)

    # Assert
    assert isinstance(result, str)  # Should return a base64 string or URL
    assert len(result) > 0
    mock_client_instance.models.generate_images.assert_called_once()
    
    call_kwargs = mock_client_instance.models.generate_images.call_args.kwargs
    assert call_kwargs.get("model") == "gemini-3.1-flash-image-preview"
    
    # Verify aesthetic enforcement in the prompt
    prompt_used = str(call_kwargs.get("prompt", "")).lower()
    assert scene_description.lower() in prompt_used
    assert any(keyword in prompt_used for keyword in ["pixel art", "geocities", "neon", "techy", "16-bit"])


@patch("lib.storybook_generator.genai.Client")
def test_generate_pixel_art_illustration_empty_description(
    mock_genai_client: MagicMock
) -> None:
    """
    Tests illustration generation with an empty scene description.
    Should raise a ValueError before attempting an API call.
    """
    # Arrange
    scene_description = "   "
    mock_client_instance = mock_genai_client.return_value

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        generate_pixel_art_illustration(scene_description)
    
    assert "Scene description cannot be empty" in str(exc_info.value)
    mock_client_instance.models.generate_images.assert_not_called()


@patch("lib.storybook_generator.genai.Client")
def test_generate_pixel_art_illustration_api_failure(
    mock_genai_client: MagicMock
) -> None:
    """
    Tests that the function surfaces or handles exceptions when the Image API fails.
    """
    # Arrange
    scene_description = "A futuristic city with flying cars."
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.models.generate_images.side_effect = Exception("Image Generation Quota Exceeded")

    # Act & Assert
    with pytest.raises(Exception) as exc_info:
        generate_pixel_art_illustration(scene_description)
    
    assert "Image Generation Quota Exceeded" in str(exc_info.value)