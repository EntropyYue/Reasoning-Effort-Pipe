"""
title: Reasoning Effort Pipe
author: EntropyYue
author_url: https://github.com/EntropyYue
funding_url: https://github.com/EntropyYue/Reasoning-Effort-Pipe
version: 0.1.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator, List
import os
from openai import OpenAI


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="Effort",
            description="Prefix to be added before model names.",
        )
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Base URL for accessing OpenAI API endpoints.",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="API key for authenticating requests to the OpenAI API.",
        )
        THINK_TIMES: int = Field(
            default=3,
            description="Number of times to think before responding.",
        )
        WAIT_WORDS: str = Field(
            default="Wait, ",
            description="Words to be used while waiting.",
        )
        MODEL_WHITELIST: List[str] = Field(
            default=[],
            description="List of allowed model IDs. Leave empty to allow all models.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.client = OpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            base_url=self.valves.OPENAI_API_BASE_URL,
        )

    def pipes(self):
        models = [m.id for m in self.client.models.list().data]
        # Apply whitelist filtering
        if self.valves.MODEL_WHITELIST:
            models = [m for m in models if m in self.valves.MODEL_WHITELIST]
        return [{"id": m, "name": f"{self.valves.NAME_PREFIX} / {m}"} for m in models]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        model_id = body["model"][body["model"].find(".") + 1 :]

        # Check model against whitelist
        if self.valves.MODEL_WHITELIST and model_id not in self.valves.MODEL_WHITELIST:
            return f"Error: Model '{model_id}' is not allowed."

        payload = body
        payload["model"] = model_id
        payload["stop"] = "</think>"
        payload["messages"].append(
            {
                "role": "assistant",
                "content": "",
            }
        )

        for count in range(self.valves.THINK_TIMES + 2):
            try:
                response = self.client.chat.completions.create(
                    **payload,
                )
                for chunk in response:
                    payload["messages"][-1]["content"] = (
                        payload["messages"][-1]["content"]
                        + chunk.choices[0].delta.content
                    )
                    yield chunk.choices[0].delta.content

                if count < self.valves.THINK_TIMES:
                    payload["messages"][-1]["content"] = (
                        payload["messages"][-1]["content"] + self.valves.WAIT_WORDS
                    )
                    yield self.valves.WAIT_WORDS

                else:
                    payload["stop"] = None

            except Exception as e:
                return f"Error: {e}"
