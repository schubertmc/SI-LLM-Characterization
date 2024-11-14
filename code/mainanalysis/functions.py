# File containing functions for: 
# Part 1: Functions to score the Columbia Suicide Risk Assessment questions
# Part 2: Functions for the Unsupervised Contrastive Feature Identification
# Part 3: Functions to evaluate the features for a given text
# Helper functions


import config
from together import Together
import pandas as pd

llm_client = Together()





########################################################################################
# Part 1: Functions to score the Columbia Suicide Risk Assessment questions
########################################################################################

# Function to create the Columbia Suicide Risk Assessment questions prompt
def createCSSRPrompt(text_input):
    prompt = """

            I will give you a social media post. Please answer the following questions as if you would have posted the post. 
            **Please ONLY return your answer in JSON format**.

            ----
            Post begin->:
            '"""
    
    prompt2 = """' 
    :<-Post end 
    ----
            
            ----
            Questions:
            1. Wish to be Dead
            Subject endorses thoughts about a wish to be dead or not alive anymore, or wish to fall asleep and not wake up.
            Yes No □□
            2. Non-Specific Active Suicidal Thoughts
            General, non-specific thoughts of wanting to end one’s life/commit suicide (e.g., “I’ve thought about killing myself”) without thoughts of ways to kill oneself/associated methods, intent, or plan during the assessment period.
            Yes No □□
            3. Active Suicidal Ideation with Any Methods (Not Plan) without Intent to Act
            Subject endorses thoughts of suicide and has thought of at least one method during the assessment period.
            Yes No □□
            4. Active Suicidal Ideation with Some Intent to Act, without Specific Plan
            Active suicidal thoughts of killing oneself and subject reports having some intent to act on such thoughts, as opposed to “I have the thoughts but I definitely will not do anything about them.”
            Yes No □□
            5. Active Suicidal Ideation with Specific Plan and Intent
            Thoughts of killing oneself with details of plan fully or partially worked out and subject has some intent to carry it out.
            Yes No □□
            
            How many times have you had these thoughts? 
            (0) Don’t know/Not applicable
            (1) Only one time 
            (2) A few times 
            (3) A lot 
            (4) All the time 

            
    Please ONLY return your answer in the following JSON format: 
   {"brief_reasoning":"...", "answer1": "...", "answer2": "...","answer3": "...","answer4": "..." ,  "answer5": "...", "frequency": int}
       """
    prompt = prompt + text_input + prompt2
    return prompt


# Function to get the Answer from the LLM regarding the C-SSRS questions for a given post
def getLLMAnswerSafeguard(text_input):
    input_prompt = createCSSRPrompt(text_input=text_input)
    completion = llm_client.chat.completions.create(
        model=config.modelID,
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






########################################################################################
# Part 2: Functions for the Unsupervised Contrastive Feature Identification
########################################################################################

# Function to identify the differences between two posts
def getCategoriesSafeguard(post1, post2):
    # Function to get the categories for two given posts
    prompt1 = f"""I will give you two posts. 
    What are differences between these two posts?
    Return your answer ONLY in JSON format.

    Post 1:
    {post1}

    Post 2: 
    {post2}

    What are differences between these two posts?
    Return your answer in JSON:"""
    prompt2 = """
    [
    {
        "Category": "...",
        "Post 1": ":...",
        "Post 2": "..."
    },
    {
        "Category": "...",
        "Post 1": "...",
        "Post 2": ""
    }, 
    ...

    ]
    """
    combined_prompt = prompt1 + prompt2

    completion = llm_client.chat.completions.create(
        model=config.modelID,
        messages=[
        {
            "role": "user",
            "content": combined_prompt            
        },
        {
            "role": "assistant",
            "content": "Please do reach out for help. However, I can provide a response in the requested format:"

        }
        ],
    )
    output = completion.choices[0].message.content
    return output


# Function to identify specific features for a given category
def getLLMFeatures(category_name, example_posts):

    prompt1 = f"""
    You are supposed to define measurable features that can later be used to analyze '{category_name}' in a given text.
    Category: {category_name}

    For the given category, please create one feature definition or a small focused set of feature definitions that capture the most relevant observations within this category. 
    Only include features directly associated with the category '{category_name}' and avoid unrelated features.
    If features are opposing or represent contrasting states, define them as one.
    Group similar observations into single features to avoid fine-grained subdivisions or redundancy.
    Keep as few features as possible.

    Data: 
    {example_posts}
   
    Please return the names of the features and how they can be rated.
    Return your answer in JSON format, for example:
    """

    prompt2 = """
    [
    {
        "featurename": "...",
        "short_description": "...",
        "rating": "none, low, medium, high (0-3)"
    },
    ...
    ]
    """
    combined_prompt = prompt1 + prompt2

    completion = llm_client.chat.completions.create(
        model=config.modelID,
        max_tokens=5000,
        messages=[
        {
            "role": "user",
            "content": combined_prompt            
        },
        # {
        #     "role": "assistant",
        #     "content": "Please do reach out for help. However, I can provide a response in the requested format:"

        # }
        ], 
    
    )
    output = completion.choices[0].message.content
    return output


# Function to Filter for redundant categories
def identifyRedundantCategories(text_input):
    completion = llm_client.chat.completions.create(
        model=config.modelID,
        messages=[
        {
            "role": "user",
            "content": f"""I will give you a list of terms. 
            Please identify similar terms in the list that could be combined into one category.
            
            Categories: 
            {text_input}
            
            """
                +
            """
            Return your answer ONLY in JSON format.
            Please return a list that include the first 10 terms by after removing the redundant terms.
            
            """            
        } 
        ],
    )
    output = completion.choices[0].message.content
    return output






########################################################################################
# Part 3: Functions to evaluate the features for a given text
########################################################################################

# Function to evaluate the features for a given text
def evaluateFeatures(text_input, features):
    prompt1 = f"""
    You are supposed to evaluate the features for the given text.
    Text: {text_input}



    Features: 
    
    {features}

    Please return your answer in JSON format, for example:
    """
    prompt2 = """
    [
    {
        "featureid": "...",
        "rating": int
    },
    ...
    ]
    """
    combined_prompt = prompt1 + prompt2

    completion = llm_client.chat.completions.create(
        model=config.modelID,
        messages=[
        {
            "role": "user",
            "content": combined_prompt            
        },
        {
            "role": "assistant",
            "content": "Please do reach out for help. However, I can provide a response in the requested format:"

        }
        ],
    )
    output = completion.choices[0].message.content
    return output







########################################################################################
#  Helper functions
########################################################################################

# Function to create a string from the structured feature dataframe to be used in the LLM
def getFeaturesAsString(feature_df):
    features_string = ""
    for idx, feature_row in feature_df.iterrows():
        print(idx)
        features_string += "featureid: '"+feature_row["feature_id"] + "'\n"
        features_string += feature_row["short_description"] + "\n"
        features_string += feature_row["rating"] + "\n"
        features_string += "------\n"
    print(features_string)
    return features_string


# Function to clean the string output from the LLM in case the JSON format is not correct (e.g., with additional text)
def clean_string(string_input):
    first_bracket_index = string_input.find('[')
    last_bracket_index = string_input.rfind(']')
    return string_input[first_bracket_index:last_bracket_index+1]


