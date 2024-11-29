"""
This file handles the logic for answer generation given a prompt as input to the LLM.
    ##TODO Just plugging and playing existing state of the art LLMs
"""

def generate_answer_with_llm(prompt_with_context: str) -> str:
    """
    Generates an answer using an LLM given a prompt and the additional
    context provided. If we want to test with other LLMs we should change
    the method name to reflect the LLM we are using.

    Args:
        prompt_with_context (str): String containing the input prompt 
            with additional contextual information.

    Returns:
        str: The generated response from the LLM based on the input.
    """
    pass