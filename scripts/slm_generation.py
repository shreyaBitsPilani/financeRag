import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class SLMResponseGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes a T5-Small model and tokenizer.
        You can switch to other open-source models, e.g. 't5-base', 'facebook/bart-base', etc.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        # Decide whether to use GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_response(self, query, retrieved_docs, max_length=512):
        """
        Generate either a short summary or a precise numeric answer
        depending on the query.
        """
        # Combine top documents into a context string
        context = ""
        for i, doc in enumerate(retrieved_docs[:3]):
            context += f"Document {i+1} [Re-rank Score: {doc['re_rank_score']}]: {doc['text']}\n"

        # Modify the prompt to avoid unnecessary output and directly ask for the numeric value and parameter
        prompt = (
            f"You are a helpful assistant that answers questions concisely based on the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Instructions:\n"
            "1. Return the **Value** and **Parameter** from the document with the highest re-rank score that best answers the question.\n"
            "2. **Do not include any other context, details, or metadata. Just provide the numeric value and parameter.**\n"
        )


        if ((any(word in query.lower() for word in ['two', '2', 'multiple', 'over']) and 'years' in query.lower()) or ('2023' in query.lower() and '2024' in query.lower())):
            prompt = (
                "You are a helpful assistant that produces a descriptive summary based on provided context. "
                "The context contains data with a 'value', 'parameter', and 'year'. "
                "Your task is to aggregate and summarize the values and parameters by year. "
                "For each year, produce a clear, complete sentence that states the total or summarized value along with its parameter. "
                "Make sure that the summary is coherent and only includes the aggregated results as per the available years. \n\n"
                f"Question: {query}\n\n"
                "Context:\n"
                f"{context}\n\n"
                "Instructions:\n"
                "1. Parse the provided context to identify all entries with their respective value, parameter, and year.\n"
                "2. Aggregate the values and parameters for each year if multiple entries exist.\n"
                "3. For each year, write a full sentence that describes the aggregated result in a natural, clear language.\n"
                "4. Do not include any extra information besides these sentences.\n"
                )



        # Debugging - Check the generated prompt
        # print(prompt)

        # Ensure the prompt is processed as intended
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

        # Decode the output, and clean it to return only relevant content
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Strip unwanted characters and ensure it contains only the value and parameter
        answer = answer.strip().split('\n')[0]  # Only return the first line with the answer
        print(answer)
        return answer



if __name__ == "__main__":
    model = SLMResponseGenerator()

    # Example usage 1: A numeric query
    question1 = "Total Revenue for the year 2023"
    docs_numeric = [
        {"text": "Year: 2023, Parameter: Total Revenue, Value: 383285000000.0"},
        {"text": "Year: 2024, Parameter: Total Revenue, Value: 391035000000.0"}
    ]
    answer1 = model.generate_response(question1, docs_numeric)
    print("Q1:", question1)
    print("A1:", answer1, "\n")

    # Example usage 2: A more general/explanatory query
    question2 = "Summarize the revenue trends for 2023 and 2024"
    docs_summary = [
        {"text": "Year: 2023, Parameter: Total Revenue, Value: 383285000000.0"},
        {"text": "Year: 2024, Parameter: Total Revenue, Value: 391035000000.0"},
        {"text": "The revenue shows an increase from 2023 to 2024 based on the given values."}
    ]
    answer2 = model.generate_response(question2, docs_summary)
    print("Q2:", question2)
    print("A2:", answer2)
