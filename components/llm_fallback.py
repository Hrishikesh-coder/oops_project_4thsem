import os
from openai import OpenAI
from typing import Optional

class LLMExtractionFallback:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"): # Update base_url depending on your Llama 3.3 provider
        """
        Initializes the LLM fallback agent.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url 
        )
        self.model = "llama-3.3-70b-versatile" # Replace with your provider's specific Llama 3.3 model string

    def extract_and_clean(self, raw_noisy_text: str) -> Optional[str]:
        """
        Passes messy or incomplete PDF text to Llama 3.3 for structured extraction.
        """
        system_prompt = (
            "You are an expert data extraction agent. Your task is to take noisy, "
            "poorly formatted text extracted from a PDF and reconstruct it into "
            "clean, accurate, and structured plain text. Do not hallucinate information. "
            "Only output the cleaned text, without conversational filler."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Clean and extract the following text:\n\n{raw_noisy_text}"}
                ],
                temperature=0.1, # Keep it low for extraction tasks to reduce hallucination
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[LLM Fallback Error] Extraction failed: {e}")
            return None