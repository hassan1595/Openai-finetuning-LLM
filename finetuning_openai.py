from openai import OpenAI
import openai
from data import DataTweets
import json
import os
from io import StringIO
import pandas as pd
import base64
import argparse

class FinetuneOpenai:
    """
    Class for fine-tuning OpenAI models using preprocessed customer support data.
    """

    def __init__(self, 
                 finetune_file=os.path.join("chatgpt_training", "finetune_data_chatgpt_4o_mini.jsonl"), 
                 validate_file=os.path.join("chatgpt_training", "validate_data_chatgpt_4o_mini.jsonl"), 
                 metrics_file=os.path.join("metrics", "step_metrics.csv"), 
                 finetune_batch_size=32):
        """
        Initializes the fine-tuning process with file paths and batch size.

        Parameters:
        - finetune_file: Path to the JSONL file for fine-tuning data.
        - validate_file: Path to the JSONL file for validation data.
        - metrics_file: Path to save fine-tuning metrics.
        - finetune_batch_size: Batch size for fine-tuning.
        """
        self.finetune_file = finetune_file
        self.finetune_batch_size = finetune_batch_size
        self.validate_file = validate_file
        self.metrics_file = metrics_file
        self.openai_key = ""
        self.dt = DataTweets()

    def get_system_prompt(self):
        """
        Generates the system prompt for fine-tuning, detailing the task objective.

        Returns:
        - system_prompt: A formatted string describing the task for the fine-tuned model.
        """
        all_companies = self.dt.get_all_companies()

        system_prompt = f"""
        You are an AI model for assigning customer support inquiries from X (formerly Twitter) to the correct company. Analyze the input, which is a batch of conversations provided as a list of lists. Each inner list contains messages in chronological order. Identify the company associated with each conversation.
        Your task is to infer the company based on context such as language, keywords, products, or services. Output one of the following companies: {all_companies}. If unsure or if the company cannot be inferred, output "Unknown".
        Inputs include conversations or standalone requests. Names have been anonymized as "Person". Inputs may be in multiple languages.
        Prioritize accuracy. Prefer "Unknown" over incorrect guesses. Do not let placeholders like "Person" influence your inference.
        Output only a (python) list of strings where each string is the company name or "Unknown".
        """
        return system_prompt

    def __prepare_data__(self):
        """
        Prepares and writes fine-tuning and validation data to JSONL files.
        """
        # Get balanced train and test datasets
        convs_train_balanced, companies_train_balanced, convs_test_balanced, companies_test_balanced = self.dt.get_convs_train_test()
        system = self.get_system_prompt()

        # Write fine-tuning data
        with open(self.finetune_file, "w") as file:
            for idx in range(0, len(convs_train_balanced), self.finetune_batch_size):
                convs_batch = convs_train_balanced[idx:idx+self.finetune_batch_size]
                companies_batch = companies_train_balanced[idx:idx+self.finetune_batch_size]

                batch_line = {"messages": [{"role": "system", "content": system}, 
                                           {"role": "user", "content": str(convs_batch)}, 
                                           {"role": "assistant", "content": str(companies_batch)}]}
                file.write(json.dumps(batch_line) + "\n")

        # Write validation data
        with open(self.validate_file, "w") as file:
            for idx in range(0, len(convs_test_balanced), self.finetune_batch_size):
                convs_batch = convs_test_balanced[idx:idx+self.finetune_batch_size]
                companies_batch = companies_test_balanced[idx:idx+self.finetune_batch_size]

                batch_line = {"messages": [{"role": "system", "content": system}, 
                                           {"role": "user", "content": str(convs_batch)}, 
                                           {"role": "assistant", "content": str(companies_batch)}]}
                file.write(json.dumps(batch_line) + "\n")

    def finetune(self):
        """
        Performs the fine-tuning process on OpenAI's platform.
        """
        # Initialize OpenAI client
        client = OpenAI(api_key=self.openai_key)

        # Prepare data if not already present
        if not os.path.exists(self.finetune_file) or not os.path.exists(self.validate_file):
            self.__prepare_data__()

        # Upload the training file
        training_file_response = client.files.create(
            file=open(self.finetune_file, "rb"),
            purpose="fine-tune"
        )
        training_file_id = training_file_response.id  # Extract the training file ID

        # Upload the validation file
        validation_file_response = client.files.create(
            file=open(self.validate_file, "rb"),
            purpose="fine-tune"
        )
        validation_file_id = validation_file_response.id  # Extract the validation file ID

        # Create the fine-tuning job
        res = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model="gpt-4o-mini-2024-07-18",
            hyperparameters={
                "n_epochs": 3
            }
        )

        # Print the response
        print(res)

    def status(self, download=True):
        """
        Checks the status of the fine-tuning job and downloads metrics if available.

        Parameters:
        - download: Whether to download the metrics file if the fine-tuning is successful.
        
        Returns:
        - fine_tuned_model: Name of the fine-tuned model if successful, otherwise None.
        """
        # Initialize OpenAI client
        client = OpenAI(api_key=self.openai_key)

        # Fetch the list of fine-tuning jobs
        try:
            jobs = client.fine_tuning.jobs.list().data
        except Exception as e:
            print(f"Error fetching fine-tuning jobs: {e}")
            return

        # Check if there are any jobs
        if not jobs:
            print("No fine-tuning jobs found.")
            return

        # Find the most recent job
        most_recent_job = max(jobs, key=lambda job: job.created_at)
        job_id = most_recent_job.id

        # Retrieve details of the most recent fine-tuning job
        try:
            job_details = client.fine_tuning.jobs.retrieve(job_id)
            print(f"Most Recent Fine-Tuning Job:\n{job_details}")
        except Exception as e:
            print(f"Error retrieving fine-tuning job details: {e}")
            return

        # Check the job status
        if job_details.status in ["running", "pending"]:
            print("Fine-tuning job is still in progress. Fetching events...")
            try:
                events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id).data
                if events:
                    print("Latest Event:")
                    print(events[-1])  # Print the most recent event
                else:
                    print("No events found for this job yet.")
            except Exception as e:
                print(f"Error fetching fine-tuning events: {e}")
        elif job_details.status == "succeeded":
            print("Fine-tuning job has completed successfully!")
            if download:
                content = client.files.content(job_details.result_files[0])
                decoded_content = base64.b64decode(content.read()).decode("utf-8")
                csv_file = StringIO(decoded_content)
                df = pd.read_csv(csv_file)
                df.to_csv(self.metrics_file, index=False)
                print(f"Metrics file downloaded as {self.metrics_file}")
            return job_details.fine_tuned_model
        elif job_details.status == "failed":
            print("Fine-tuning job failed.")
            if job_details.error:
                print(f"Error Details: {job_details.error}")

        return None


def main():
    """
    Main function to execute the script based on provided arguments.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Manage fine-tuning tasks for OpenAI models.")
    parser.add_argument(
        "-s", "--status",
        action="store_true",
        help="Check the status of the fine-tuning job."
    )
    parser.add_argument(
        "-ft", "--finetune",
        action="store_true",
        help="Start the fine-tuning process."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of FinetuneOpenai
    ft = FinetuneOpenai()

    # Handle the provided arguments
    if args.status:
        ft.status()
    elif args.finetune:
        ft.finetune()
    else:
        print("Please provide a valid option: -s (status) or -ft (finetune).")

if __name__ == "__main__":
    main()
