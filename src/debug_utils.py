#!/usr/bin/env python3
"""
Debug script to inspect LLM messages before they're sent to Bedrock.
This helps identify the "list index out of range" error.
"""

import json
import sys
from pathlib import Path
from typing import Any
from openhands.sdk.llm import LLM
import litellm
from .utils import batch_inference

# Monkey patch the LLM completion to log messages
def patch_llm_for_debugging(path: Path):
    """Patches the LLM class to log all messages before sending to Bedrock."""
    original_transport_call = LLM._transport_call

    def debug_transport_call(
        self,
        *,
        messages,
        enable_streaming,
        on_token,
        **kwargs,):
        """Wrapper that logs messages before calling the original method."""

        # Also save to file for later inspection
        debug_file = path / "debug_llm_messages.json"
        with open(debug_file, "w") as f:
            messages_serializable = []
            for msg in messages:
                if hasattr(msg, 'model_dump'):
                    messages_serializable.append(msg.model_dump())
                else:
                    messages_serializable.append(msg)

            json.dump({
                "model": self.model,
                "num_messages": len(messages),
                "messages": messages_serializable,
                "kwargs": {k: str(v) for k, v in kwargs.items()}
            }, f, indent=2, default=str)

        # Call the original method
        return original_transport_call(self, 
            messages=messages,
            enable_streaming=enable_streaming,
            on_token=on_token,
            **kwargs)

    LLM._transport_call = debug_transport_call

def replay_messages(path: str, replay_n: int = 1):
    """Replays the LLM messages from the given debug file."""
    with open(path, 'r') as f:
        data = json.load(f)

    model = data['model']
    messages = data['messages']
    for k in data['kwargs']:
        data['kwargs'][k] = eval(data['kwargs'][k])

    def batch_inference_wrapper(model, messages, kwargs):
        response = litellm.completion(
            model=model,
            messages=messages,
            **kwargs
        )
        print("\n✅ SUCCESS!")
        print(f"Response: {response.choices[0].message.content}")
        return [response]


    batch_inference(
        program=batch_inference_wrapper,
        args_list=[{
            "model": model,
            "messages": messages,
            "kwargs": data['kwargs'],
        } for _ in range(replay_n)],
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_utils.py <debug_file_path> [<replay_n>]")
        sys.exit(1)

    debug_file_path = sys.argv[1]
    replay_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    replay_messages(debug_file_path, replay_n)