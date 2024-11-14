# config.py
import dotenv
import os

dotenv.load_dotenv(".env")

# Configuration settings
data_path = "../MEDplus_hackathon/data/"
modelID = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
train_data_path = os.path.join(data_path, "swmh_train.jsonl")
output_path = "../data"

