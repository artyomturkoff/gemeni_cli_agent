import os
import argparse

from dotenv import load_dotenv
from google import genai
from google.genai import types
from promts import system_prompt

from functions.call_function import available_functions, call_function



def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY found")
    client = genai.Client(api_key=api_key)

    parser = argparse.ArgumentParser(description="Gemini Bot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]

    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=messages,
        config=types.GenerateContentConfig(tools=[available_functions],
                                            system_instruction=system_prompt, 
                                            temperature=0),
        )
    if not response.usage_metadata:
        raise RuntimeError("No response metadata")
    
    if args.verbose:    
        print(f"User prompt: {args.user_prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
    

    if not response.function_calls:
        print("Response:")
        print(response.text)
        return
    
    function_responses: list[types.Part] = []
    for function_call in response.function_calls:
        result = call_function(function_call, args.verbose)
        if (
            not result.parts
            or not result.parts[0].function_response
            or not result.parts[0].function_response.response
        ):
            raise RuntimeError(f"Empty function response for {function_call.name}")
        if args.verbose:
            print(f"-> {result.parts[0].function_response.response}")
        function_responses.append(result.parts[0])



if __name__ == "__main__":
    main()
