import torch
from transformers import AutoTokenizer, pipeline


class AnswerGenerator:
    """
    Handles the logic for generating answers using multiple Large Language Models (LLMs).
    The class evaluates each model's suitability based on predefined metrics and selects the best model dynamically.
    
    How to use in the experiment:
    
    answer_generator = AnswerGenerator()
    short_prompt = (
        "Table: Revenue (in millions) | 2022 | 2021\n"
        "Revenue | $500 | $450\n"
        "Profit | $50 | $45\n"
        "Question: What was the revenue growth rate from 2021 to 2022?\n"
        "[Final Answer]: "
    )
    response = answer_generator.generate_answer_with_llm(len(short_prompt.split()), short_prompt)
    """

    def __init__(self):
        """
        Initializes the LLMs and their configurations for short and medium contexts.
        """
        # Define LLM configurations for short and medium contexts
        self.short_context_models = [
            {"name": "meta-llama/LLaMA-3-70b-chat", "max_tokens": 5000},
            {"name": "meta-llama/LLaMA-3.1", "max_tokens": 5000},
            {"name": "qwen/Qwen2-72B-Instruct", "max_tokens": 5000},
        ]
        self.medium_context_models = [
            {"name": "qwen/Qwen1.5-32B-Chat", "max_tokens": 25000},
            {"name": "qwen/Qwen2-72B-Instruct", "max_tokens": 25000},
        ]
        
        #TODO: find open source llms for long context

        self.models = {}  # Dictionary to store initialized pipelines
        self.initialize_models()

    def initialize_models(self):
        """
        Initializes pipelines for all configured models.
        """
        for model_config in self.short_context_models + self.medium_context_models:
            model_name = model_config["name"]
            print(f"Loading model: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
            )

    def evaluate_model(self, model_name, prompt: str, hallucination_index_threshold: float = 0.1) -> float:
        """
        Evaluates the model's output based on a hallucination index.
        In this example, the hallucination index is simulated for demonstration.

        Args:
            model_name (str): The name of the model to evaluate.
            prompt (str): The input prompt.
            hallucination_index_threshold (float): Maximum allowable hallucination index.

        Returns:
            float: Simulated hallucination index (lower is better).
        """
        #TODO: Simulated evaluation logic; replace with actual metrics if available
        print(f"Evaluating model {model_name} for hallucination...")
        hallucination_index = 0.05  # Simulate low hallucination index for now
        return hallucination_index

    def select_best_model(self, context_length: int, prompt: str) -> str:
        """
        Selects the best model based on the context length and hallucination index.

        Args:
            context_length (int): Length of the input context in tokens.
            prompt (str): The input prompt.

        Returns:
            str: The name of the best model for the given context and prompt.
        """
        if context_length <= 5000:
            candidates = self.short_context_models
        else:
            candidates = self.medium_context_models
        #TODO: add long context when available

        # Evaluate all candidates and select the best based on hallucination index
        # Simple version, looking for better ways to do it
        best_model = None
        best_score = float("inf")
        for model_config in candidates:
            model_name = model_config["name"]
            hallucination_index = self.evaluate_model(model_name, prompt)
            if hallucination_index < best_score:
                best_model = model_name
                best_score = hallucination_index

        print(f"Selected best model: {best_model} with hallucination index: {best_score}")
        return best_model

    def generate_answer_with_llm(self, context_length: int, prompt_with_context: str) -> str:
        """
        Generates an answer using the best LLM for the given context.

        Args:
            context_length (int): Length of the input context in tokens.
            prompt_with_context (str): The input prompt containing the question and any relevant context.

        Returns:
            str: The generated answer from the selected LLM.
        """
        best_model_name = self.select_best_model(context_length, prompt_with_context)
        pipeline = self.models[best_model_name]

        try:
            response = pipeline(
                prompt_with_context,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            return response[0]["generated_text"]
        except Exception as e:
            print(f"Error generating answer with model {best_model_name}: {e}")
            return "An error occurred while generating the answer."
        
