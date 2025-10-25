import anthropic
from dotenv import load_dotenv
import os
import json

load_dotenv()


def get_lm_output(prompt):
    """Get the output from the LM, returns a JSON array of objects with the schema described below
    {
        "native": boolean,
        "action": string,
        "generative": boolean,
        "prompt": string
    }"""
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    SYSTEM_PROMPT = f"""You are an intent normalizer for voice-to-image editing.

    Task:
    Given a natural-language instruction, output a JSON array of objects with this exact schema:

    native: boolean

    action: string (present only if native=true), formatted exactly as one of:
    "(resize, (m,n))" where m,n are integers (pixels)
    "(rotate, deg)" where deg ∈ {0, 90, 180, 270} (normalize words like “ninety” → 90). 
    "(grayscale, )"
    "(vflip, )"
    "(hflip, )"
    "(sharpness, s)" where s is a number
    "(saturation, x)" where x is a number

    generative: boolean

    prompt: string (present only if generative=true), a concise diffusion prompt describing the edit for that step

    Native vs Generative:

    Native (and only these): resize, rotate, grayscale, vflip, hflip, sharpness, saturation.

    Everything else is Generative: add/remove/replace objects, recolor specific objects or backgrounds, style/material changes, relighting, segmentation/inpainting, background swaps, etc.

    Rules:

    Split multi-step instructions into atomic steps in the spoken order.
    
    For rotate, only include the degree if it is one of the following: {0, 90, 180, 270}. Otherwise, ignore the rotate command completely and do not include it in the JSON array.

    For native steps: native=true, generative=false, fill action, prompt="".

    For generative steps: generative=true, native=false, action="", fill prompt (short, specific, preserve scene unless told otherwise).

    Normalize numbers/units (e.g., “ninety degrees” → 90; “1920 by 1080” → (1920,1080)).

    Remember: Only the seven native actions listed are native. Everything else is generative. Output JSON only."""

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    raw_output = message.content[0].text
    cleaned_json_string = raw_output.strip().removeprefix("```json").strip().removesuffix("```").strip() if raw_output else None
    json_array = json.loads(cleaned_json_string) if cleaned_json_string else None
    return json_array

if __name__ == "__main__":
    print(get_lm_output("bla bla bla bla bla "))