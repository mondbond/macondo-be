import requests
from langchain.llms.base import LLM
from typing import Optional, List

# Not used currently, but could be extended for more features
class OllamaClient(LLM):
    model: str = "mistral:instruct"
    base_url: str = "http://localhost:11434"
    verbose: bool = True

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        if hasattr(prompt, "to_string"):
          prompt = prompt.to_string()

        print("=====OLLAMA PROMPT=====")
        print(prompt)
        print("========================")

        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.1),
        }
        response = requests.post(f"{self.base_url}/v1/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "completions" in data and len(data["completions"]) > 0:
            return data["completions"][0].get("text", "")

        response = data.get("choices", "")[0].get("text", "No response")

        print("=====OLLAMA RESPONSE=====")
        print(response)
        print("========================")

        return response

    @property
    def _identifying_params(self):
        return {"model": self.model, "base_url": self.base_url, "verbose": self.verbose}

    @property
    def _llm_type(self) -> str:
        return "ollama" + self.model

    def invoke(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        return self._call(prompt, stop=stop, **kwargs)
