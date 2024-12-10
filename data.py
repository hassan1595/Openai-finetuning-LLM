import pandas as pd
import math
import re
from sklearn.model_selection import train_test_split
import random
import os
from preprocess import Preprocessing
from tqdm import tqdm

class DataTweets:
    """
    A class to handle the processing of tweet data for machine learning tasks. 

    Attributes:
    - train_data_file: Path to the training dataset CSV file.
    - target_size: Target size for each class in the training dataset.
    - test_target_size: Target size for each class in the test dataset.
    - prod_data_file: Path to the production dataset CSV file.
    """

    def __init__(self, train_data_file=os.path.join("datasets", "training_data.csv"), 
                 target_size=200, test_target_size=20, 
                 prod_data_file=os.path.join("datasets", "prod_data_cleaned_new.csv")):
        """
        Initializes the DataTweets object with file paths and target sizes.

        Parameters:
        - train_data_file: Path to the training dataset CSV file.
        - target_size: Target size for each class in the training dataset.
        - test_target_size: Target size for each class in the test dataset.
        - prod_data_file: Path to the production dataset CSV file.
        """
        self.train_data_file = train_data_file
        self.test_target_size = test_target_size
        self.target_size = target_size
        self.prod_data_file = prod_data_file

    def __balance_dataset__(self, conversations, companies, target_size, shuffle=False, random_state=None):
        """
        Balances the dataset by ensuring all classes are resized to the target size
        (either by truncating or oversampling) and optionally shuffles the dataset.

        Parameters:
        - conversations: List of conversation texts.
        - companies: List of company labels corresponding to conversations.
        - target_size: The desired size for each class.
        - shuffle: Whether to shuffle the balanced dataset.
        - random_state: Seed for the random generator to ensure reproducibility.

        Returns:
        - Balanced (and optionally shuffled) lists of conversations and companies.
        """
        # Set the random seed if provided
        if random_state is not None:
            random.seed(random_state)

        # Combine conversations and companies into tuples
        data = list(zip(conversations, companies))
        
        # Group data by company
        grouped_data = {}
        for conv, company in data:
            if company not in grouped_data:
                grouped_data[company] = []
            grouped_data[company].append(conv)

        # Balance the dataset
        balanced_data = []
        for company, convs in grouped_data.items():
            if len(convs) < target_size:
                # Oversample if class size is smaller than target size
                oversampled = random.choices(convs, k=target_size - len(convs))
                convs += oversampled
            elif len(convs) > target_size:
                # Truncate if class size exceeds target size
                convs = convs[:target_size]

            # Add the balanced conversations and their company labels
            balanced_data.extend((conv, company) for conv in convs)

        # Shuffle the balanced dataset if required
        if shuffle:
            random.shuffle(balanced_data)

        # Separate conversations and companies
        balanced_conversations, balanced_companies = zip(*balanced_data)

        return list(balanced_conversations), list(balanced_companies)

    def __extract_mentions_with_regex__(self, text):
        """
        Extracts all mentions (e.g., @username) from the given text and 
        returns a list of mentions and the text without the mentions.

        Parameters:
        - text: A string containing tweet text.

        Returns:
        - mentions: A list of extracted mentions.
        - text_without_mentions: The text with mentions removed.
        """
        mentions_pattern = r"@\w+"  # Regex to match mentions
        mentions = re.findall(mentions_pattern, text)  # Extract mentions
        text_without_mentions = re.sub(mentions_pattern, "", text).strip()  # Remove mentions from text
        return mentions, text_without_mentions

    def __apply_preprocess__(self, data):
        """
        Applies preprocessing to a nested list of data while maintaining the original structure.

        Parameters:
        - data: A nested list where each sublist contains strings.

        Returns:
        - Preprocessed data with the same nested structure.
        """
        shape_data = [len(d) for d in data]  # Record the size of each sublist
        flatten_data = [item for sublist in data for item in sublist]  # Flatten the nested list
        p = Preprocessing(flatten_data)  # Preprocessing instance
        new_flatten_data = p.apply()  # Apply preprocessing
        new_data = []
        idx = 0
        for length in shape_data:  # Restore nested structure
            new_data.append(new_flatten_data[idx: idx + length])
            idx += length
        return new_data

    def get_all_companies(self):
        """
        Retrieves all companies from the training dataset, including an 'Unknown' placeholder.

        Returns:
        - A set of all unique company IDs in the dataset.
        """
        df = pd.read_csv(self.train_data_file)  # Load training dataset
        df_c = df[df["inbound"] == False]  # Filter for outbound messages
        all_companies = set(df_c["author_id"])  # Collect unique author IDs
        all_companies.add("Unknown")  # Add a placeholder for unknown companies
        return all_companies

    def get_convs_train_test(self):
        """
        Creates balanced train and test datasets with preprocessed conversations.

        Returns:
        - convs_train_balanced: Balanced training conversations.
        - companies_train_balanced: Balanced company labels for training data.
        - convs_test_balanced: Balanced test conversations.
        - companies_test_balanced: Balanced company labels for test data.
        """
        # Load and clean training dataset
        df = pd.read_csv(self.train_data_file)
        df["visited"] = False
        df['response_tweet_id'] = df['response_tweet_id'].map(
            lambda x: x.split(",")[0] if isinstance(x, str) and "," in x else x)
        df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].map(
            lambda x: x.split(",")[0] if isinstance(x, str) and "," in x else x)
        df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce')
        df = df.dropna(subset=['tweet_id'])
        df.set_index("tweet_id", drop=False, inplace=True)
        df['response_tweet_id'] = pd.to_numeric(df['response_tweet_id'], errors='coerce')
        df['in_response_to_tweet_id'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')

        # Process conversations and identify companies
        all_companies = self.get_all_companies()
        convs = []
        companies = []
        for tweet_id in tqdm(df["tweet_id"], "Creating Training Conversations"):
            if not df.loc[tweet_id]["visited"]:
                conv = []
                company = None
                current = tweet_id
                parent = df.loc[current]["in_response_to_tweet_id"]
                # Traverse the conversation thread upwards
                while not math.isnan(parent) and parent in df.index and not df.loc[parent]["visited"]:
                    current = parent
                    parent = df.loc[current]["in_response_to_tweet_id"]

                # Traverse the thread downwards
                while not math.isnan(current) and current in df.index:
                    df.at[current, "visited"] = True
                    mentions, text_without_mentions = self.__extract_mentions_with_regex__(df.loc[current]["text"])
                    conv.append(text_without_mentions)
                    if not company:
                        for mention in mentions:
                            if mention[1:] in all_companies:
                                company = mention[1:]
                        if not df.loc[current]["inbound"]:
                            company = df.loc[current]["author_id"]
                    current = df.loc[current]["response_tweet_id"]
                    if current not in df["tweet_id"]:
                        current = float("nan")

                convs.append(conv)
                companies.append(company if company else "Unknown")

        # Apply preprocessing and balance datasets
        convs = self.__apply_preprocess__(convs)
        assert len(convs) == len(companies)

        convs_train, convs_test, companies_train, companies_test = train_test_split(
            convs, companies, test_size=0.1, random_state=42)

        convs_train_balanced, companies_train_balanced = self.__balance_dataset__(
            convs_train, companies_train, self.target_size, shuffle=True, random_state=42)
        convs_test_balanced, companies_test_balanced = self.__balance_dataset__(
            convs_test, companies_test, self.test_target_size, shuffle=True, random_state=42)

        return convs_train_balanced, companies_train_balanced, convs_test_balanced, companies_test_balanced

    def get_convs_prod(self):
        """
        Extracts and preprocesses production conversations for inference.

        Returns:
        - df_original: The original production DataFrame.
        - convs: Preprocessed production conversations.
        - tweet_ids_convs: Lists of tweet IDs corresponding to each conversation.
        """
        # Load and clean production dataset
        df = pd.read_csv(self.prod_data_file)
        df_original = df.copy()
        df["visited"] = False
        df['text'] = df['text'].map(lambda x: x if isinstance(x, str) else "")
        df['response_tweet_id'] = df['response_tweet_id'].map(
            lambda x: x.split(",")[0] if isinstance(x, str) and "," in x else x)
        df['in_response_to_tweet_id'] = df['in_response_to_tweet_id'].map(
            lambda x: x.split(",")[0] if isinstance(x, str) and "," in x else x)
        df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce')
        df = df.dropna(subset=['tweet_id'])
        df.set_index("tweet_id", drop=False, inplace=True)
        df['response_tweet_id'] = pd.to_numeric(df['response_tweet_id'], errors='coerce')
        df['in_response_to_tweet_id'] = pd.to_numeric(df['in_response_to_tweet_id'], errors='coerce')

        # Initialize containers for conversations and tweet IDs
        convs = []
        tweet_ids_convs = []
        
        # Traverse threads to create conversations
        for tweet_id in tqdm(df["tweet_id"], "Creating Production Conversations Step 1"):
            if not df.loc[tweet_id]["visited"]:
                conv = []
                tweet_ids_conv = []
                current = tweet_id
                parent = df.loc[current]["in_response_to_tweet_id"]

                while not math.isnan(parent) and parent in df.index and not df.loc[parent]["visited"]:
                    current = parent
                    parent = df.loc[current]["in_response_to_tweet_id"]

                while not math.isnan(current) and current in df.index:
                    df.at[current, "visited"] = True
                    conv.append(df.loc[current]["text"])
                    tweet_ids_conv.append(current)
                    current = df.loc[current]["response_tweet_id"]
                
                convs.append(conv)
                tweet_ids_convs.append(tweet_ids_conv)

        # Handle unprocessed conversations
        for tweet_id in tqdm(df["tweet_id"], "Creating Production Conversations Step 2"):
            if not df.loc[tweet_id]["visited"]:
                current = tweet_id
                while not math.isnan(current)and current in df.index:
                    df.at[current, "visited"] = True
                    conv.append(df.loc[current]["text"])
                    tweet_ids_conv.append(current)
                    current = df.loc[current]["response_tweet_id"]
                
                
                
                convs.append(conv)
                tweet_ids_convs.append(tweet_ids_conv)

        # Apply preprocessing and validate
        convs = self.__apply_preprocess__(convs)
        assert len(convs) == len(tweet_ids_convs)

        return df_original, convs, tweet_ids_convs
