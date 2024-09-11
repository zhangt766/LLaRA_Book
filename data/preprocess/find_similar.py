#!/usr/bin/env python3

import pickle
from typing import Optional
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse

def load_model_and_tokenizer(model_name_or_path):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, output_hidden_states=True)
    except RuntimeError as e:
        if "Failed to import transformers.models.llama.modeling_llama" in str(e):
            print("Encountered Flash Attention compatibility issue. Attempting to load without Flash Attention...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                output_hidden_states=True,
                use_flash_attention_2=False
            )
        else:
            raise
    try:
        model.to(device)
    except:
        print("Warning: Failed to move model to GPU. Using CPU instead.")
        device = torch.device("cpu")
        model.to(device)
    print(f"Model loaded on {device}.")
    return tokenizer, model, device

def load_data(file_path):
    return pd.read_pickle(file_path)

def load_movie_dict(item_file):
    item_df = pd.read_csv(item_file, sep='::', header=None, encoding='latin-1', engine='python', usecols=[0, 1])
    item_df.columns = ['movie_id', 'movie_title']
    movie_dict = dict(zip(item_df['movie_id'], item_df['movie_title']))
    return movie_dict

def map_movie_names_only(seq, movie_dict):
    return [movie_dict[id] if id in movie_dict else id for id in seq]

def extract_sequences(df, movie_dict):
    df['movie_names_only'] = df['seq'].apply(lambda x: map_movie_names_only(x, movie_dict))
    return df

def get_movie_embeddings(movie_list, tokenizer, model, device):
    embeddings = []
    max_length = 512
    for movies in tqdm(movie_list):
        movie_string = " ".join(str(movie) for movie in movies)
        inputs = tokenizer(movie_string, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            movie_embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu()
        embeddings.append(movie_embedding)
    return torch.stack(embeddings)

def get_topk_similar_indices(similarity_scores, topK):
    indices = np.argsort(-np.array(similarity_scores.to(torch.float32)))
    topk_indices = np.ones((indices.shape[0], topK))
    for i, indice in enumerate(indices):
        tmp = indice[indice != i]
        topk_indices[i] = tmp[:topK]
    return topk_indices

def get_topK_candidate(df, tokenizer, model, device, topK=10):
    embeddings = get_movie_embeddings(df['movie_names_only'].tolist(), tokenizer, model, device)
    similarity_scores = embeddings @ embeddings.T
    most_similar_indices = np.array(get_topk_similar_indices(similarity_scores, topK)).tolist()
    df['most_similar_seq_index'] = [json.dumps(most_similar_idxs) for most_similar_idxs in most_similar_indices]
    
    def safe_get_seq(idxs):
        return [df.loc[int(idx), 'seq'] if int(idx) in df.index else None for idx in json.loads(idxs)]
    
    df['most_similar_seq'] = df['most_similar_seq_index'].apply(safe_get_seq)
    df['most_similar_seq'] = df['most_similar_seq'].apply(lambda x: [seq for seq in x if seq is not None])
    
    return df

def add_most_similar_seq_next(df, movie_dict):
    def safe_get_next(idxs):
        return [df.loc[int(idx), 'next'] if int(idx) in df.index else None for idx in json.loads(idxs)]
    
    df['most_similar_seq_next'] = df['most_similar_seq_index'].apply(safe_get_next)
    df['most_similar_seq_next'] = df['most_similar_seq_next'].apply(lambda x: [item for item in x if item is not None])
    
    df['most_similar_seq_name'] = df['most_similar_seq'].apply(lambda seqs: [[movie_dict.get(item, "Unknown") for item in items] for items in seqs])
    df['most_similar_seq_next_name'] = df['most_similar_seq_next'].apply(lambda nexts: [movie_dict.get(item, "Unknown") for item in nexts])
    return df

def process_data(file_path, item_file, output_file_path, model_name_or_path):
    tokenizer, model, device = load_model_and_tokenizer(model_name_or_path)
    data = load_data(file_path)
    movie_dict = load_movie_dict(item_file)
    df = extract_sequences(data, movie_dict)
    df = get_topK_candidate(df, tokenizer, model, device)
    df = add_most_similar_seq_next(df, movie_dict)
    df.to_pickle(output_file_path)
    return df

def main():
    parser = argparse.ArgumentParser(description="Process movie data for recommendation system")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--item_file", type=str, required=True, help="Path to the item file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output file")
    
    args = parser.parse_args()
    
    processed_df = process_data(args.data_path, args.item_file, args.output_path, args.model_path)
    print("Data processing completed. Results saved to:", args.output_path)

if __name__ == "__main__":
    main()