from together import Together
import dotenv
import jinja2
import pandas as pd
import json

dotenv.load_dotenv(".env")
llm_client = Together()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Create a prompt from template and text input
# Template: prompts/cssrs.j2
def createCSSRSPrompt(text_input):
    templateLoader = jinja2.FileSystemLoader(searchpath="./prompts/")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "cssrs.j2"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(text_input=text_input)
    return outputText

# Create a prompt from template, text input and feature definitions
# Template: prompts/features.j2
def createFeaturePrompt(text_input, features):
    templateLoader = jinja2.FileSystemLoader(searchpath="./prompts/")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "features.j2"
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(text_input=text_input, features=features)
    return outputText

# Ask the model to evaluate the prompt
def promptModel(input_prompt):
    completion = llm_client.chat.completions.create(
        model=model_id,
        messages=[
        {
            "role": "user",
            "content": input_prompt            
        },
        {
            "role": "assistant",
            "content": "Please do reach out for help. However, I can provide a response in the requested format:"

        }
        ],
    )
    output = completion.choices[0].message.content
    return output

def evaluateCSSRS(text_input):
    # Create a prompt from template and text input
    input_prompt = createCSSRSPrompt(text_input=text_input)

    # Ask the model to evaluate the prompt
    output = promptModel(input_prompt)

    # Parse the output from the model
    model_output = json.loads(output)

    # Convert Yes/No answers to True/False
    result = {}
    result["answer1"] = model_output["answer1"] == "Yes"
    result["answer2"] = model_output["answer2"] == "Yes"
    result["answer3"] = model_output["answer3"] == "Yes"
    result["answer4"] = model_output["answer4"] == "Yes"
    result["answer5"] = model_output["answer5"] == "Yes"
    result["freq"] = int(model_output["frequency"])

    return result

def evaluateFeatures(text_input):
    # Load feature definitions from features.csv
    features = pd.read_csv("features.csv")
    
    # Keep only feature_name, description, and rating_options to be provided to the LLM in CSV format
    features_definition_csv = features[["feature_name", "description", "rating_options"]].to_csv(index=False)

    # Create a prompt from template, text input and feature definitions
    input_prompt = createFeaturePrompt(text_input=text_input, features=features_definition_csv)

    # Ask the model to evaluate the prompt
    output = promptModel(input_prompt)

    # Parse the output from the model
    model_output = json.loads(output)

    # model_output = "{'relationships_Social_Isolation': 3, 'tone_Emotional_tone': 3, 'tone_Uncertainty': 3,...}"

    # Convert the model output to pandas DataFrame
    model_output_df = pd.DataFrame(model_output.items(), columns=["feature_name", "score"])

    # Merge the model output with the features
    features = pd.merge(features, model_output_df, on="feature_name")

    # Calculate average score ("score") for each cluster (group by "cluster")
    clusters = features.groupby("cluster")["score"].mean().round(2).reset_index()

    # Combine the features and clusters into a dictionary
    # result:
    #     features:
    #         relationships_Social_Isolation: 3
    #         tone_Emotional_tone: 3
    #         ...
    #     cluster:
    #         cluster1: 3.5
    #         cluster2: 2.5
    #         ...
    result = {}
    result["features"] = features.set_index("feature_name").to_dict()["score"]
    result["cluster"] = clusters.set_index("cluster").to_dict()["score"]

    return result