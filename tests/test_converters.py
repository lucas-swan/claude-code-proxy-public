"""Unit tests for request and response converters."""

import json
import pytest
from unittest.mock import MagicMock

from src.models.claude import (
    ClaudeMessagesRequest,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeSystemContent,
    ClaudeTool,
)
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
)
from src.api.endpoints import _estimate_tokens
from src.core.model_manager import ModelManager


# --- Helper fixtures ---

class MockConfig:
    big_model = "gpt-4o"
    middle_model = "gpt-4o"
    small_model = "gpt-4o-mini"
    min_tokens_limit = 100
    max_tokens_limit = 4096


mock_model_manager = ModelManager(MockConfig())


def make_request(**overrides):
    defaults = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [ClaudeMessage(role="user", content="Hello")],
    }
    defaults.update(overrides)
    return ClaudeMessagesRequest(**defaults)


# --- Request converter tests ---

class TestConvertClaudeToOpenAI:
    """Tests for convert_claude_to_openai."""

    def test_basic_text_message(self):
        req = make_request()
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o"  # sonnet -> middle_model
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["stream"] is False

    def test_system_message_string(self):
        req = make_request(system="You are helpful.")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_system_message_list(self):
        req = make_request(system=[ClaudeSystemContent(type="text", text="Part 1")])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Part 1"

    def test_model_mapping_haiku(self):
        req = make_request(model="claude-3-haiku-20240307")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o-mini"

    def test_model_mapping_sonnet(self):
        req = make_request(model="claude-3-5-sonnet-20241022")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o"

    def test_model_mapping_opus(self):
        req = make_request(model="claude-3-opus-20240229")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o"

    def test_model_passthrough_gpt(self):
        req = make_request(model="gpt-4o")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "gpt-4o"

    def test_model_passthrough_deepseek(self):
        req = make_request(model="deepseek-chat")
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["model"] == "deepseek-chat"

    def test_max_tokens_clamping(self):
        from src.core.config import config
        req = make_request(max_tokens=1)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["max_tokens"] == config.min_tokens_limit

        req = make_request(max_tokens=99999)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["max_tokens"] == config.max_tokens_limit

    def test_image_conversion(self):
        req = make_request(messages=[
            ClaudeMessage(role="user", content=[
                ClaudeContentBlockText(type="text", text="What is this?"),
                ClaudeContentBlockImage(type="image", source={
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgo="
                }),
            ])
        ])
        result = convert_claude_to_openai(req, mock_model_manager)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_tool_use_conversion(self):
        req = make_request(
            messages=[
                ClaudeMessage(role="user", content="Check the weather"),
                ClaudeMessage(role="assistant", content=[
                    ClaudeContentBlockToolUse(
                        type="tool_use",
                        id="tool_123",
                        name="get_weather",
                        input={"location": "NYC"}
                    )
                ]),
                ClaudeMessage(role="user", content=[
                    ClaudeContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="tool_123",
                        content="Sunny, 72F"
                    )
                ]),
            ],
            tools=[ClaudeTool(
                name="get_weather",
                description="Get weather",
                input_schema={"type": "object", "properties": {"location": {"type": "string"}}}
            )]
        )
        result = convert_claude_to_openai(req, mock_model_manager)

        # Check tools
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

        # Check assistant message with tool_calls
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["tool_calls"][0]["id"] == "tool_123"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

        # Check tool result message
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "tool_123"
        assert tool_msg["content"] == "Sunny, 72F"

    def test_tool_choice_auto(self):
        req = make_request(tool_choice={"type": "auto"})
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_specific(self):
        req = make_request(tool_choice={"type": "tool", "name": "get_weather"})
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "get_weather"

    def test_top_p_and_stop_sequences(self):
        req = make_request(top_p=0.9, stop_sequences=["END"])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["top_p"] == 0.9
        assert result["stop"] == ["END"]

    def test_thinking_not_passed_through(self):
        """thinking is accepted by the model but should not appear in OpenAI request."""
        from src.models.claude import ClaudeThinkingConfig
        req = make_request(thinking=ClaudeThinkingConfig(enabled=True))
        result = convert_claude_to_openai(req, mock_model_manager)
        assert "thinking" not in result

    def test_top_k_not_passed_through(self):
        """top_k is accepted by the model but should not appear in OpenAI request."""
        req = make_request(top_k=50)
        result = convert_claude_to_openai(req, mock_model_manager)
        assert "top_k" not in result

    def test_empty_content_user_message(self):
        req = make_request(messages=[ClaudeMessage(role="user", content=None)])
        result = convert_claude_to_openai(req, mock_model_manager)
        assert result["messages"][0]["content"] == ""

    def test_empty_content_assistant_message(self):
        req = make_request(messages=[
            ClaudeMessage(role="user", content="Hi"),
            ClaudeMessage(role="assistant", content=None),
        ])
        result = convert_claude_to_openai(req, mock_model_manager)
        assistant_msg = result["messages"][1]
        assert assistant_msg["content"] is None


# --- Response converter tests ---

class TestConvertOpenAIToClaudeResponse:
    """Tests for convert_openai_to_claude_response (non-streaming)."""

    def test_basic_text_response(self):
        req = make_request()
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls_response(self):
        req = make_request()
        openai_resp = {
            "id": "chatcmpl-456",
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "call_abc"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"location": "NYC"}

    def test_max_tokens_stop_reason(self):
        req = make_request()
        openai_resp = {
            "id": "chatcmpl-789",
            "choices": [{"message": {"content": "Truncated..."}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["stop_reason"] == "max_tokens"

    def test_empty_choices_raises(self):
        req = make_request()
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            convert_openai_to_claude_response({"choices": []}, req)
        assert exc_info.value.status_code == 500

    def test_no_content_returns_empty_text(self):
        req = make_request()
        openai_resp = {
            "id": "chatcmpl-000",
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_model_preserved_from_request(self):
        req = make_request(model="claude-3-haiku-20240307")
        openai_resp = {
            "id": "chatcmpl-111",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["model"] == "claude-3-haiku-20240307"

    def test_invalid_tool_arguments_handled(self):
        req = make_request()
        openai_resp = {
            "id": "chatcmpl-222",
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "my_func", "arguments": "invalid json{"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = convert_openai_to_claude_response(openai_resp, req)
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["input"] == {"raw_arguments": "invalid json{"}


# --- Token estimation tests ---

class TestEstimateTokens:
    """Tests for _estimate_tokens helper."""

    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_english_text(self):
        # ~4 chars per token
        result = _estimate_tokens("Hello world test")
        assert result >= 1

    def test_chinese_text(self):
        # ~1.5 chars per token for CJK
        result = _estimate_tokens("你好世界测试")
        assert result >= 1

    def test_mixed_text(self):
        result = _estimate_tokens("Hello 你好 world 世界")
        assert result >= 1

    def test_single_char(self):
        assert _estimate_tokens("a") == 1

    def test_cjk_char(self):
        assert _estimate_tokens("一") == 1
