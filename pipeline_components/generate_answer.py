from dexter.llms.llm_engine_orchestrator import LLMEngineOrchestrator
from transformers import AutoTokenizer, pipeline
import torch

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

    def __init__(self, tiny_version: bool = False):
        """
        Initializes the LLMs and their configurations for short and medium contexts.
        """
        # Define LLM configurations for short and medium contexts
        self.short_context_models = [
            {"name": "meta-llama/Llama-3.1-405B-Instruct", "max_tokens": 5000},
        ]
        self.medium_context_models = [
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "max_tokens": 25000},
        ]
        self.long_context_models = [
            {"name": "starsy/Llama-3-70B-Instruct-Gradient-262k-AWQ", "max_tokens": 200000},
        ]
        self.tiny_context_models = [
            {"name": "meta-llama/Llama-2-7b-chat-hf", "max_tokens": 1000},
        ]

        self.models = {}  # Dictionary to store initialized pipelines
        self.initialize_models(tiny_version=tiny_version)

    def initialize_models(self, tiny_version: bool = False):
        """
        Initializes instances for all configured models.
        """
        # Use LLMEngineOrchestrator for initialization
        config_instance = LLMEngineOrchestrator()
            
        models = self.tiny_context_models if tiny_version else self.short_context_models + self.medium_context_models + self.long_context_models
        
        for model_config in models:
            model_name = model_config["name"]
            print(f"Loading model: {model_name}...")
            
            llm_instance = config_instance.get_llm_engine(
                data="",
                llm_class="llama",  # Adjust based on the engine used
                model_name=model_name,
            )
            self.models[model_name] = llm_instance
            
            #Prev version to test if the orchestrator from dexter doesn't work
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # self.models[model_name] = pipeline(
            #     "text-generation",
            #     model=model_name,
            #     tokenizer=tokenizer,
            #     device_map="auto",
            #     torch_dtype=torch.float16,
            # )

    def evaluate_model(self, model_name, query: str, context: str, hallucination_index_threshold: float = 0.1) -> float:
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

    def select_best_model(self, context_length: int, query: str, context: str) -> str:
        """
        Selects the best model based on the context length and hallucination index.

        Args:
            context_length (int): Length of the input context in tokens.
            query (str): The input question or query.
            context (str): The context or passages containing relevant information.

        Returns:
            str: The name of the best model for the given context and prompt.
        """
        if context_length <= 5000:
            candidates = self.short_context_models
        elif context_length <= 25000:
            candidates = self.medium_context_models
        else:
            candidates = self.long_context_models
        

        # Evaluate all candidates and select the best based on hallucination index
        # Simple version, looking for better ways to do it
        best_model = None
        best_score = float("inf")
        for model_config in candidates:
            model_name = model_config["name"]
            hallucination_index = self.evaluate_model(model_name, query, context)
            if hallucination_index < best_score:
                best_model = model_name
                best_score = hallucination_index

        print(f"Selected best model: {best_model} with hallucination index: {best_score}")
        return best_model

    def generate_answer_with_llm(self, context_length: int, query: str, context: str) -> str:
        """
        Generates an answer using the best LLM for the given context.

        Args:
            context_length (int): Length of the input context in tokens.
            query (str): The input question or query.
            context (str): The context or passages containing relevant information.

        Returns:
            str: The generated answer from the selected LLM.
        """
        # Add system and user prompts for better guidance
        system_prompt = (
            "Follow the given examples and, based on the context provided, "
            "output the final answer to the question using the information in the context. "
            "Respond in the form: [Final Answer]: \n"
        )
        user_prompt = (
            "[Question]: When does monsoon season end in the state the area code 575 is located?\n"
            "[Final Answer]: mid-September.\n"
            "[Question]: What is the current official currency in the country where Ineabelle Diaz is a citizen?\n"
            "[Final Answer]: United States dollar.\n"
            "[Question]: Where was the person who founded the American Institute of Public Opinion in 1935 born?\n"
            "[Final Answer]: Jefferson.\n\n"
            f"Follow the above examples, and given the context below, "
            f"answer the question:\n [Question]: {query}\n [Context]: {context}\n"
        )
        
        best_model_name = self.select_best_model(context_length, query, context)
        llm_instance = self.models[best_model_name]

        try:
            # Generate the answer using the selected model
            response = llm_instance.get_chat_completion(user_prompt, system_prompt)
            if "not possible" in response.lower() or "unknown" in response.lower():
                return "[Final Answer]: Unable to provide an answer with the given context."
            elif "[Final Answer]:" in response:
                return response.split("[Final Answer]:", 1)[1].strip()
            else:
                return "[Final Answer]: Could not parse a valid answer."
        except Exception as e:
            print(f"Error generating answer with model {best_model_name}: {e}")
            return "[Final Answer]: An error occurred while generating the answer."
    
        # Prev version to try if dexter doesn't work
        # try:
        #     response = pipeline(
        #         prompt_with_context,
        #         max_new_tokens=256,
        #         do_sample=True,
        #         temperature=0.7,
        #         top_k=50,
        #         top_p=0.95,
        #     )
        #     return response[0]["generated_text"]
        # except Exception as e:
        #     print(f"Error generating answer with model {best_model_name}: {e}")
        #     return "An error occurred while generating the answer."
        
