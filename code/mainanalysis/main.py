import os
import json
import pandas as pd
import random
from datetime import datetime

# Importing configuration and functions
import config
from functions import createCSSRPrompt, getLLMAnswerSafeguard, getLLMFeatures, getFeaturesAsString, clean_string


# Load data path from config
train_data_path = config.train_data_path

# Load data and add unique identifier for each post, downsample 
train_dataset = pd.read_json(train_data_path, lines=True)
train_dataset["ID"] = "ID" + train_dataset.index.astype(str) +"_"+ train_dataset["label"]
train_dataset = train_dataset.sample(10000)
train_dataset.to_csv(os.path.join(config.output_path, "dataset_train_10k.csv"))


########################################################################################
# Part 1: Evaluate C-SSRS questions
########################################################################################

# Loop through the dataset and evaluate the CSSRS questions
train_dataset["output"] = None
train_dataset["CSSRS_1"] = None
train_dataset["CSSRS_2"] = None
train_dataset["CSSRS_3"] = None
train_dataset["CSSRS_4"] = None
train_dataset["CSSRS_5"] = None
train_dataset["CSSRS_freq"] = None
train_dataset["brief_reasoning"] = None
train_dataset["error"] = None
count = 0

for i, row in train_dataset.iterrows():
    print(count, i)
    count += 1
    text_input = row["text"]
    label = row["label"]
    output = getLLMAnswerSafeguard(text_input)
    train_dataset.at[i, "output"] = output

    try:
        output = json.loads(output)
        train_dataset.at[i, "brief_reasoning"] = output["brief_reasoning"]
        train_dataset.at[i, "CSSRS_1"] = output["answer1"]
        train_dataset.at[i, "CSSRS_2"] = output["answer2"]
        train_dataset.at[i, "CSSRS_3"] = output["answer3"]
        train_dataset.at[i, "CSSRS_4"] = output["answer4"]
        train_dataset.at[i, "CSSRS_5"] = output["answer5"]
        train_dataset.at[i, "CSSRS_freq"] = output["frequency"]
    except Exception as e: 
        print(i)
        print(str(e))
        print(output)
        train_dataset.at[i, "error"] = str(e)

# Add column to indicate if any of the CSSRS questions is "Yes"
train_dataset["CSSRS_any"] = train_dataset[["CSSRS_1", "CSSRS_2", "CSSRS_3", "CSSRS_4", "CSSRS_5"]].apply(lambda x: 1 if "Yes" in x.values else 0, axis=1)
train_dataset.to_csv(os.path.join(config.output_path, "dataset_train_10k_CSSRS_ratings.csv"))

# Filter for posts with suicidal ideation, save dataset
si_dataset = train_dataset[train_dataset["CSSRS_any"] ==1]
si_dataset = si_dataset.reset_index(drop=True)
si_dataset.to_csv(os.path.join(config.output_path, "si_dataset_v1.csv"))





########################################################################################
# Part 2: Identify Features with Unsupervised Contrastive Feature Identification
########################################################################################

#---------------------#
# Identify Differences
#---------------------#

# Task: Compare the different ways in which each post expresses suicidal ideation.
# First sample 200 unique pairs of rows, then compare the ways in which each post expresses suicidal ideation.

# Sample 200 unique pairs of rows
sampled_indices = random.sample(range(1, len(si_dataset.index)), 200)
sampled_pairs = [(sampled_indices[i], sampled_indices[i+100]) for i in range(100)]

# Save sampled_pairs to a file
sampled_pairs_file = os.path.join(config.output_path, "sampled_pairs.json")
with open(sampled_pairs_file, "w") as f:
    json.dump(sampled_pairs, f)
 
# Now compare the ways in which each post expresses suicidal ideation, please list any differences you notice.
categories = []
for idx, pair in enumerate(sampled_pairs):
    print(idx, pair)
    post1 = train_dataset.iloc[pair[0]]
    post2 = train_dataset.iloc[pair[1]]

    post1_text = post1["text"]
    post2_text = post2["text"]
    output = getCategoriesSafeguard(post1_text, post2_text)
    print(output)

    try:
        loaded = json.loads(output)
        categories.append(loaded)
        print("Loading worked instantly")
    except Exception as e:
        print("Loading didnt work, try differently")
        print(str(e))

        try: 
            last_bracket_index = output.rfind(']')
            output[:last_bracket_index+1]
            loaded = json.loads(output[:last_bracket_index+1])
            categories.append(loaded)
            print("Loading worked after cutting")
        except Exception as e2:
            print("Cutting didnt work!")
            print(str(e))


# Flatten the nested list of categories
flattened_categories = [item for sublist in categories for item in sublist]

# Save flattened categories as a DataFrame
categories_df = pd.DataFrame(flattened_categories)
categories_df["Category"] = categories_df["Category"].str.lower()
categories_comparison_file = os.path.join(config.output_path, "categories_comparisons.csv")
categories_df.to_csv(categories_comparison_file)


#---------------------#
# Select top 10 categories
#---------------------#
# Get top 10 redundant categories using identifyRedundantCategories function

# Calculate category counts and convert to a list of category names
vals = categories_df["Category"].value_counts()
output = identifyRedundantCategories(str(vals.index.tolist()))
# Save filtered top 10 categories to JSON file
filtered_categories_file = os.path.join(config.output_path, "filtered_top10_categories.json")
with open(filtered_categories_file, "w") as f:
    json.dump(output, f)
# Save filtered top 10 categories to JSON file
filtered_categories_file = os.path.join(config.output_path, "filtered_top10_categories.json")
with open(filtered_categories_file, "w") as f:
    json.dump(output, f)
filtered_categories = json.loads(output)
# Get the top 10 categories
target_categories = [cat["name"].lower() for cat in filtered_categories["categories"]][:10]




#---------------------#
# Create features for each category
#---------------------#

# Initialize dictionary to store feature definitions for each category
features_dict = {}
# Loop through each target category and retrieve example comparisons for feature definitions
for category in target_categories:
    print(category)
    # Filter category-specific rows from categories_df
    category_df = categories_df[categories_df["Category"] == category]
    example_posts = ""
    
    # Concatenate example post pairs for the current category
    for i, row in category_df.iterrows():
        example_posts += row["Post 1"] + "\n"
        example_posts += "------\n"
        example_posts += row["Post 2"] + "\n"
        example_posts += "------\n"
    
    # Generate feature definitions using the LLM for the current category
    output = getLLMFeatures(category, example_posts)
    features_dict[category] = output    
    print(output)  

# Clean the LLM output to ensure valid JSON format for each category's feature definitions
for key in features_dict.keys():
    print(key)
    clean = clean_string(features_dict[key])
    try:
        # Parse JSON-formatted string to Python dictionary
        loaded = json.loads(clean)
        features_dict[key] = loaded
        print(loaded)
    except Exception as e:
        print(str(e))  # Log any errors in JSON parsing






#---------------------#
# Reformat extracted features into structured dataframes
#---------------------#

# Initialize a list to store dataframes for each category's feature definitions
df_list = []

# Convert each category's feature definitions into a structured dataframe
for key in features_dict.keys():
    print(key)
    try:
        # Load the current category's feature list into a DataFrame
        cur_list = features_dict[key]
        cur_df = pd.DataFrame.from_dict(cur_list)
        
        # Add the category as a new column to each row in the current DataFrame
        cur_df["category"] = key
        
        # Relocate the 'category' column to the first position for better readability
        cols = cur_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        cur_df = cur_df[cols]
        
        # Append the current DataFrame to the list
        df_list.append(cur_df)
    except Exception as e:
        print(str(e))  # Log any errors encountered during DataFrame creation
        print(key)
        print(features_dict[key])



#---------------------#
# Definition of final feature dataframe
#---------------------#


# Concatenate all dataframes into a single DataFrame
final_df = pd.concat(df_list)
final_df["featurename"].value_counts()
final_df["category"] = final_df["category"].str.replace(" ", "")
final_df["category"].value_counts()
final_df["feature_id"] = final_df["category"] + "_" + final_df["featurename"]
final_df["feature_id"] = final_df["feature_id"].str.replace(" ", "_")
final_df["feature_id"].tolist()
final_df.to_csv(f"{config.output_path}/feature_definitions_final.csv")





########################################################################################
# Part 3: Evaluation of features
########################################################################################

#---------------------#
# Prepare features for providing it to the LLM
#---------------------#


# Split final_df into a list of DataFrames for each category
category_dfs = {}
for category in target_categories:
    category_dfs[category] = final_df[final_df["category"] == category.replace(" ", "")]
    print(category_dfs[category].shape)

# Build feature strings for each category to be used in the LLM
features_strings_dict = {category: getFeaturesAsString(category_dfs[category]) for category in category_dfs.keys()}

# Sample 1000 rows from si_dataset and save to CSV
si_dataset_1k = si_dataset.sample(1000)
si_dataset_1k.to_csv(f"{config.output_path}/si_dataset_1k.csv")






#---------------------#
# Evaluate all features
#---------------------#

# Loop through posts in si_dataset for feature evaluation
evaluated_feature_dfs = []
count = 0
for idx, row in si_dataset.iterrows():
    start = datetime.now()
    
    # Retrieve post details
    text_input = row["text"]
    id = row["ID"]
    print(idx, id, count)
    count += 1

    feature_evaluation_dict = {}

    # Loop through different feature categories
    for category, feat_string in features_strings_dict.items():
        # Evaluate features for the specific category
        output = evaluateFeatures(text_input, feat_string)
        
        # Parse and clean LLM output
        try:
            feat = json.loads(clean_string(output))
            feature_evaluation_dict[category] = feat
        except Exception as e:
            print(f"Error in category {category}: {e}")
            print("Output:", output)

    # Create a dictionary for current post's evaluated features
    clean_dict = {"ID": id}
    for category, features in feature_evaluation_dict.items():
        for feature in features:
            clean_dict[feature["featureid"]] = feature["rating"]

    # Append the evaluated features to the DataFrame list
    post_df = pd.DataFrame.from_dict(clean_dict, orient='index').T
    evaluated_feature_dfs.append(post_df)
    end = datetime.now()
    print(f"Evaluation time: {end - start}")

# Concatenate all evaluated feature DataFrames and save as CSV
bound = pd.concat(evaluated_feature_dfs)
bound.to_csv(f"{config.output_path}/evaluated_features_f1000.csv", index=False)
