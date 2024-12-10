from openai import OpenAI
from data import DataTweets
from finetuning_openai import FinetuneOpenai
import sys
import os
import ast
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd

class Apply:
    """
    Class for applying a fine-tuned model to predict companies for customer support conversations and analyzing the results.
    """

    def __init__(self, 
                 output_file=os.path.join("output", "output.csv"), 
                 analyze_plot_file=os.path.join("output", "analyze.png"), 
                 apply_batch_size=8):
        """
        Initializes the Apply class, loading the fine-tuned model and necessary components.

        Parameters:
        - output_file: Path to save the prediction results as a CSV file.
        - analyze_plot_file: Path to save the analysis plot as an image.
        - apply_batch_size: Batch size for querying the model.
        """
        self.ft = FinetuneOpenai()  # Instance of the fine-tuning class
        self.dt = DataTweets()  # Instance of the data processing class
        self.output_file = output_file
        self.apply_batch_size = apply_batch_size
        self.analyze_plot_file = analyze_plot_file
        self.finetuned_model = self.ft.status(download=False)  # Retrieve the fine-tuned model name
        if not self.finetuned_model:
            print("No Fine-Tuned Model found")
            sys.exit()

    def apply(self):
        """
        Applies the fine-tuned model to predict companies for customer support conversations.
        Saves the results to a CSV file.
        """
        client = OpenAI(api_key=self.ft.openai_key)  # Initialize OpenAI client
        df, convs_prod, tweet_ids_convs_prod = self.dt.get_convs_prod()  # Retrieve production data
        df.set_index("tweet_id", inplace=True, drop=False)
        df["company"] = "Unknown"  # Initialize company predictions as "Unknown"

        # Process conversations in batches
        for idx in tqdm(range(0, len(convs_prod), self.apply_batch_size), "Querying the LLM"):
            convs_batch = convs_prod[idx:idx+self.apply_batch_size]
            tweet_ids_convs_batch = tweet_ids_convs_prod[idx:idx+self.apply_batch_size]

            system = self.ft.get_system_prompt()  # Generate system prompt
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": str(convs_batch)}
            ]
            # Use the fine-tuned model for inference
            response = client.chat.completions.create(
                model=self.finetuned_model,
                messages=messages
            )
            res = response.choices[0].message.content  # Extract model response

            try:
                companies = ast.literal_eval(res)  # Safely evaluate response to Python list
                if len(companies) != self.apply_batch_size:
                    continue

                # Map predicted companies to corresponding tweet IDs
                for tweet_ids_convs, company in zip(tweet_ids_convs_batch, companies):
                    for tweet_id in tweet_ids_convs:
                        if df.loc[tweet_id]["company"] == "Unknown":
                            df.at[tweet_id, "company"] = company

            except (ValueError, SyntaxError):
                # Handle invalid responses from the model
                continue

        df.to_csv(self.output_file, index=False)  # Save results to CSV

    def analyze(self):
        """
        Analyzes the predicted results and generates a frequency plot of companies.

        Saves the plot as an image.
        """
        if not os.path.exists(self.output_file):  # Check if predictions file exists
            return

        df = pd.read_csv(self.output_file)  # Load predictions from CSV
        frequency = df["company"].value_counts()  # Count occurrences of each company
        frequency = frequency.sort_values(ascending=False)  # Sort companies by frequency

        # Generate and save bar plot
        plt.figure(figsize=(12, 8))
        frequency.plot(kind="bar", color="lightblue", edgecolor="black")
        plt.title("Frequency of Companies (Sorted)", fontsize=16)
        plt.xlabel("Company", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(rotation=90, ha="right", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.analyze_plot_file)
        plt.show()

def main():
    """
    Main function to execute the application process and analysis.
    """
    a = Apply()
    a.apply()
    a.analyze()

if __name__ == "__main__":
    main()
