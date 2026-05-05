"""Unit tests for the retry mechanism in src.core.client."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError

from src.core.client import _retry_with_backoff


@pytest.fixture
def mock_coro_factory():
    """Create a mock coroutine factory."""
    return AsyncMock()


class TestRetryWithBackoff:
    """Test the retry mechanism with exponential backoff."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, mock_coro_factory):
        """Should return result immediately if first attempt succeeds."""
        mock_coro_factory.return_value = "success"

        result = await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert result == "success"
        assert mock_coro_factory.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_transient_error(self, mock_coro_factory):
        """Should retry on transient APIError and succeed."""
        # First two attempts fail with 500, third succeeds
        mock_coro_factory.side_effect = [
            APIError(message="Server error", request=MagicMock(), body={"status_code": 500}),
            APIError(message="Server error", request=MagicMock(), body={"status_code": 500}),
            "success"
        ]

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert result == "success"
        assert mock_coro_factory.call_count == 3
        assert mock_sleep.call_count == 2  # Slept twice (after first two failures)

    @pytest.mark.asyncio
    async def test_rate_limit_uses_longer_delay(self, mock_coro_factory):
        """Should use longer backoff for RateLimitError (429)."""
        mock_coro_factory.side_effect = [
            RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body={"error": {"type": "rate_limit_error"}}
            ),
            "success"
        ]

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert result == "success"
        assert mock_coro_factory.call_count == 2
        assert mock_sleep.call_count == 1
        # Rate limit delay: min(2^1, 8) + jitter = 2 + jitter (around 2-3s)
        delay = mock_sleep.call_args[0][0]
        assert 2.0 <= delay <= 3.0

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self, mock_coro_factory):
        """Should NOT retry on fatal errors (401, 403, 400)."""
        mock_coro_factory.side_effect = AuthenticationError(
            message="Unauthorized",
            response=MagicMock(status_code=401),
            body={"error": {"type": "authentication_error"}}
        )

        with pytest.raises(AuthenticationError):
            await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert mock_coro_factory.call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_bad_request_no_retry(self, mock_coro_factory):
        """Should NOT retry on BadRequestError (400)."""
        mock_coro_factory.side_effect = BadRequestError(
            message="Bad request",
            response=MagicMock(status_code=400),
            body={"error": {"type": "invalid_request_error"}}
        )

        with pytest.raises(BadRequestError):
            await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert mock_coro_factory.call_count == 1

    @pytest.mark.asyncio
    async def test_unknown_error_no_retry(self, mock_coro_factory):
        """Should NOT retry on unknown errors (not APIError)."""
        mock_coro_factory.side_effect = ValueError("Unknown error")

        with pytest.raises(ValueError):
            await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert mock_coro_factory.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, mock_coro_factory):
        """Should raise after max_retries attempts."""
        mock_coro_factory.side_effect = APIError(
            message="Server error",
            request=MagicMock(),
            body={"status_code": 500}
        )

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(APIError):
                await _retry_with_backoff(mock_coro_factory, max_retries=2)

        # Should try 3 times (initial + 2 retries)
        assert mock_coro_factory.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self, mock_coro_factory):
        """Should use exponential backoff: 1s, 2s, 4s..."""
        mock_coro_factory.side_effect = [
            APIError(message="Error", request=MagicMock(), body={"status_code": 500}),
            APIError(message="Error", request=MagicMock(), body={"status_code": 500}),
            APIError(message="Error", request=MagicMock(), body={"status_code": 500}),
            "success"
        ]

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await _retry_with_backoff(mock_coro_factory, max_retries=3)

        assert result == "success"
        assert mock_sleep.call_count == 3

        # Check delays are approximately 1s, 2s, 4s (with jitter 0-1)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert 1.0 <= delays[0] <= 2.0  # 1s + jitter
        assert 2.0 <= delays[1] <= 3.0  # 2s + jitter
        assert 4.0 <= delays[2] <= 5.0  # 4s + jitter

    @pytest.mark.asyncio
    async def test_backoff_capped_at_8_seconds(self, mock_coro_factory):
        """Should cap backoff at 8s + jitter (max 9s)."""
        # Trigger enough retries to reach the cap
        errors = [
            APIError(message="Error", request=MagicMock(), body={"status_code": 500})
            for _ in range(10)
        ]
        mock_coro_factory.side_effect = errors + ["success"]

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(APIError):  # Will exceed max_retries=5
                await _retry_with_backoff(mock_coro_factory, max_retries=5)

        # Check that delays don't exceed 9s (8s cap + 1s jitter)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        for delay in delays:
            assert delay <= 9.0

    @pytest.mark.asyncio
    async def test_label_in_log_messages(self, mock_coro_factory, caplog):
        """Should include label in retry log messages."""
        mock_coro_factory.side_effect = [
            APIError(message="Error", request=MagicMock(), body={"status_code": 500}),
            "success"
        ]

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock):
            import logging
            with caplog.at_level(logging.INFO):
                result = await _retry_with_backoff(
                    mock_coro_factory,
                    max_retries=3,
                    label="test_operation"
                )

        assert result == "success"
        assert "test_operation" in caplog.text


class TestRetryIntegration:
    """Integration tests for retry with real coroutine creation."""

    @pytest.mark.asyncio
    async def test_real_coroutine_factory(self):
        """Should work with real async functions."""
        call_count = 0

        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError(
                    message="Transient error",
                    request=MagicMock(),
                    body={"status_code": 500}
                )
            return "success"

        with patch('src.core.client.asyncio.sleep', new_callable=AsyncMock):
            result = await _retry_with_backoff(flaky_function, max_retries=3)

        assert result == "success"
        assert call_count == 3
