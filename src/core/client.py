import asyncio
import json
import random
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError
from src.core.config import config

# Transient upstream errors that should be retried (same ones OpenAI SDK retries,
# plus 404 which is common with load-balanced third-party APIs)
TRANSIENT_STATUS_CODES = {404, 429, 500, 502, 503, 504}

# Errors that must never be retried
FATAL_STATUS_CODES = {401, 403, 400}


def _is_transient_error(exc: APIError) -> bool:
    """Check if an API error is transient and worth retrying."""
    status = getattr(exc, 'status_code', 500)
    return status in TRANSIENT_STATUS_CODES


async def _retry_with_backoff(coro_factory, max_retries: int, label: str = ""):
    """Retry an async callable with exponential backoff + jitter for transient errors.

    coro_factory is a zero-arg callable that returns a fresh coroutine each attempt.
    Non-transient errors are raised immediately on first failure.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except (AuthenticationError, RateLimitError, BadRequestError):
            raise  # never retry client errors
        except APIError as e:
            last_exc = e
            status = getattr(e, 'status_code', 500)
            if status in FATAL_STATUS_CODES or attempt >= max_retries:
                raise
            # Exponential backoff: 1s, 2s, 4s ...  with jitter
            delay = min(2 ** attempt, 8) + random.uniform(0, 1)
            import logging
            logging.getLogger(__name__).info(
                f"Transient error {status}, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries}) {label}"
            )
            await asyncio.sleep(delay)
        except Exception:
            raise  # don't retry unknown errors
    raise last_exc  # unreachable but keeps type-checkers happy

class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.custom_headers = custom_headers or {}
        
        # Prepare default headers
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "claude-proxy/1.0.0"
        }
        
        # Merge custom headers with default headers
        all_headers = {**default_headers, **self.custom_headers}
        
        # Detect if using Azure and instantiate the appropriate client
        if api_version:
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
                timeout=timeout,
                max_retries=config.max_retries,
                default_headers=all_headers
            )
        else:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=config.max_retries,
                default_headers=all_headers
            )
        self.active_requests: Dict[str, asyncio.Event] = {}
    
    async def create_chat_completion(self, request: Dict[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support and transient error retries."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Helper: one raw API call (fresh coroutine each attempt)
            async def _do_call():
                return await self.client.chat.completions.create(**request)

            if request_id:
                # Cancellable + retried path
                retry_task = asyncio.create_task(
                    _retry_with_backoff(_do_call, config.max_retries, label="create_chat_completion")
                )
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [retry_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                if cancel_task in done:
                    retry_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")

                completion = await retry_task
            else:
                completion = await _retry_with_backoff(_do_call, config.max_retries, label="create_chat_completion")

            return completion.model_dump()

        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            if status_code in (502, 503, 504):
                raise HTTPException(
                    status_code=status_code,
                    detail=f"Upstream service temporarily unavailable (HTTP {status_code}). Please retry in a few moments."
                )
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        finally:
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support and transient error retries."""

        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event

        try:
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True

            # Create the streaming completion with retry
            streaming_completion = await _retry_with_backoff(
                lambda: self.client.chat.completions.create(**request),
                config.max_retries,
                label="create_chat_completion_stream",
            )
            
            async for chunk in streaming_completion:
                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"
            
            # Signal end of stream
            yield "data: [DONE]"
                
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False