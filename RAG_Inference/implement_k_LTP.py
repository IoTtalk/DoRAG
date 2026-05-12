import os
import shutil
import glob
import gc
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tika import parser
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from utils.tart.tart.TART.src.modeling_enc_t5 import EncT5ForSequenceClassification
from utils.tart.tart.TART.src.tokenization_enc_t5 import EncT5Tokenizer
from utils.generation import generate_with_loop
from transformers import AutoModel, AutoTokenizer
import random
from langchain.schema import Document
import requests
import time
from utils.use_ollama import query_ollama

random.seed(1207)
np.random.seed(1207)

# Configuration
config = {
    'input_text_dir': 'chunk.json',  # Directory for input text files    
    'embedding_model_path': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',  # Path to the embedding model
    'database_path': 'vectorDB_gte2_test',  # Path for saving the vector database
    'query_file': 'Qs.txt',  # Path to the query file
    'tart_model': 'facebook/tart-full-flan-t5-xl',  # Path to the TART model
    'num_results': 10,  # Number of results to retrieve
    'result_dir': 'test6',  # Directory to save results
    'result_file': 'testing2.json',
    'retrieval_method': 'retrieve_with_re_ranker',  # Choose either 'retrieve' or 'retrieve_with_re_ranker'
    'set_up_database': True,  # Whether to set up the vector database (True/False)
    'k': [1]
}
print(config)

print(f"GPU: {torch.cuda.is_available()}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyEmbedding:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir='./models'
        )
        self.model = self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir='./models')
        self.model.eval()  # 關掉 dropout 等訓練行為

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]
        attention_mask: [B, T]
        return: [B, H]
        """
        # 如果最後一個位置對於整個 batch 都是 1，代表使用 left padding
        left_padding = (attention_mask[:, -1].sum() == attention_mask.size(0))

        if left_padding:
            return hidden_states[:, -1, :]
        else:
            # right padding：找每個樣本最後一個有效 token 的索引
            last_idx = attention_mask.sum(dim=1) - 1  # [B]
            last_idx = last_idx.clamp(min=0)
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, last_idx, :]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 16  # 可依記憶體調整
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state          # [B, T, H]
                attn = inputs["attention_mask"]             # [B, T]
                batch_embeddings = self._last_token_pool(hidden, attn).cpu().tolist()
            embeddings.extend(batch_embeddings)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state              # [1, T, H]
            attn = inputs["attention_mask"]                 # [1, T]
            emb = self._last_token_pool(hidden, attn)[0]    # [H]
        return emb.cpu().tolist()


def set_vector_db(config):    
    print("Start constructing database!")
    with open(config['input_text_dir'], 'r', encoding="utf-8") as fr:
        data = json.load(fr)
    
    # Extract texts from the 'content' header of each element    
    texts = [f"{item['paper']}\n{item['section_title']}\n{item['content']}" for item in data if item.get('content') and len(item['content']) > 0]
    
    # Assuming you no longer need to split the texts    
    chunks = [Document(page_content=text) for text in texts]
    
    
    embedding_model = MyEmbedding(config['embedding_model_path'])
    print("Setting Embedding Model...")
    
    if os.path.isdir(config['database_path']):
        shutil.rmtree(config['database_path'])
    os.makedirs(config['database_path'])

    chromadb = Chroma.from_documents(chunks, 
                                     embedding=embedding_model,
                                     collection_name='coll_cosine',
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory=config['database_path'])
    chromadb.persist()
    
    print("Finish constructing database!")
    return len(chunks)


def retrieve(user_query, num, embedding_model, database_path):
    '''Basic retrieval using cosine similarity.'''
    #embedding_model = MyEmbedding(embedding_model)
    
    chromadb = Chroma(embedding_function=embedding_model,
                      collection_name='coll_cosine',
                      collection_metadata={"hnsw:space": "cosine"},
                      persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, num)
    
    unique_results = set()
    final_results = []

    for result in results:
        content = result[0].page_content
        if content not in unique_results:
            unique_results.add(content)
            final_results.append((content, result[1]))
    
    final_results.sort(key=lambda a: a[1])

    #print(final_results)
    return [[[content for content, _ in final_results]]]


def retrieve_with_re_ranker(user_query, num, embedding_model, model, tokenizer, database_path):
    '''Retrieval with re-ranking using the TART model.'''
    embedding_model = MyEmbedding(embedding_model)
    chromadb = Chroma(embedding_function=embedding_model,
                      collection_name='coll_cosine',
                      collection_metadata={"hnsw:space": "cosine"},
                      persist_directory=database_path)

    results = chromadb.similarity_search_with_score(user_query, num)
    
    unique_results = set()
    for result in results:
        unique_results.add(result[0].page_content)
    
    unique_results = list(unique_results)
    final_result = unique_results[0]
    
    in_answer = "retrieve a passage that answers this question from some paper"
    for i in range(1, len(unique_results)):
        features = tokenizer(
            ['{0} [SEP] {1}'.format(in_answer, user_query), 
             '{0} [SEP] {1}'.format(in_answer, user_query)], 
            [final_result, unique_results[i]], padding=True, truncation=True, return_tensors="pt", max_length=4096)
        with torch.no_grad():
            scores = model(**features).logits
            normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
        if np.argmax(normalized_scores) != 0:
            final_result = unique_results[i]
    #print(final_result)
    
    return [[final_result]]

def retrieve_with_re_ranker_top_k(user_query, num, k, embedding_model, model, tokenizer, database_path):
    """
    Retrieval with re-ranking using the TART model to return the top k results,
    with GPU acceleration if available.
    """
    
    # Initialize Chroma (runs on CPU) with the embedding function
    chromadb = Chroma(
        embedding_function=embedding_model,
        collection_name="coll_cosine",
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=database_path
    )

    # First-round similarity search (returns list of (Document, score))
    results = chromadb.similarity_search_with_score(user_query, num)

    # Deduplicate based on page_content
    unique_contents = []
    seen = set()
    for doc, score in results:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_contents.append(content)

    # If there are no retrieved passages, return empty list
    if not unique_contents:
        return [[]]

    # Prepare for re-ranking: compute a relevance score for each passage
    scored_results = []
    in_answer = "retrieve a passage that answers this question from some paper"

    for passage in unique_contents:
        # Build inputs for the reranker: both rows use the same passage
        texts_for_model = [
            f"{in_answer} [SEP] {user_query}",
            f"{in_answer} [SEP] {user_query}"
        ]
        passages = [passage, passage]

        features = tokenizer(
            texts_for_model,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=4096
        )

        # Move all tensors to the chosen device
        features = {name: tensor.to(device) for name, tensor in features.items()}

        # Run reranking model without gradient tracking
        with torch.no_grad():
            logits = model(**features).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()

        # Use the probability of the "relevant" class as the relevance score
        relevance_score = float(probs[0, 1])
        scored_results.append((passage, relevance_score))

    # Sort by relevance score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Extract the top k passages
    top_k_results = [item[0] for item in scored_results[:k]]

    return [[top_k_results]]

def extract_after_last_think(text):
    # Handle None or non-string input
    if not text or not isinstance(text, str):
        print("⚠️ Warning: extract_after_last_think received None or non-string input.")
        return ""

    # Find the last occurrence of </think>
    end_think_index = text.rfind("</think>")
    
    if end_think_index == -1:
        print("⚠️ Warning: No </think> tag found in the text. Returning full text.")
        return text.strip()
    
    # Extract everything after the last </think>
    after_think_text = text[end_think_index + len("</think>"):].strip()
    
    return after_think_text

def main(config):
    '''Main function to perform retrieval and generation separately.'''
    query_file = config['query_file']
    with open(query_file, 'r') as fr:
        user_queries = fr.read().split("\n")
    
    embedding_model = config['embedding_model_path']
    database_path = config['database_path']
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)    
    
    # Set up the database if required
    if config['set_up_database']:
        set_vector_db(config)
        
    embedding_model = MyEmbedding(embedding_model)
    # Retrieval step
    retrieved_results = []
    retrieved_time = []

    if config['retrieval_method'] == 'retrieve_with_re_ranker':
        model = EncT5ForSequenceClassification.from_pretrained(config['tart_model'], use_safetensors=False, cache_dir='./models')
        model.to(device)
        model.eval()
        tokenizer = EncT5Tokenizer.from_pretrained(config['tart_model'], cache_dir='./models')
        
            
    print("Starting retrieval...")
    for query in tqdm(user_queries, desc="Retrieving"):
        if config['retrieval_method'] == 'retrieve':
            torch.cuda.synchronize()
            start_time = time.time()
            result = retrieve(query, config['num_results'], embedding_model, database_path)
            torch.cuda.synchronize()
            end_time = time.time()
        elif config['retrieval_method'] == 'retrieve_with_re_ranker':
            torch.cuda.synchronize()
            start_time = time.time()            
            result = retrieve_with_re_ranker_top_k(query, config['num_results'], max(config['k']), embedding_model, model, tokenizer, database_path)
            torch.cuda.synchronize()
            end_time = time.time()
            #print(result)
        else:
            raise ValueError(f"Unknown retrieval method: {config['retrieval_method']}")
        
        retrieved_results.append(result)
        retrieved_time.append((end_time - start_time) * 1000)
        #print(retrieved_results)
    
    print("Retrieval completed.")
    
    retrieval_result_file = os.path.join(result_dir, config['result_file'])
    
    json_results = []  # List to hold all results as JSON objects

    for query_no, retrieved_result in enumerate(retrieved_results):
        json_results.append({
            "qid": query_no,
            "query": user_queries[query_no],
            "retrieved_context": retrieved_result[0][0] if retrieved_result else "No result",
            "retrieved_time": retrieved_time[query_no]
        })

    with open(retrieval_result_file, "w") as retrieval_file:
        json.dump(json_results, retrieval_file, indent=4)  # Write JSON array to file with pretty formatting
    
    del retrieved_results
    gc.collect()

    # Generation step
    print("Starting generation...")
        

    
    with open(retrieval_result_file, "r") as retrieval_file:
        retrieval_data = json.load(retrieval_file)


    for k in config['k']:    
        generated_answers = []
        generated_answers_deepseek = []
        generation_time = []
        for query_no, query in tqdm(enumerate(user_queries), desc="Generating Answers"):
            
            retrieved_content = retrieval_data[query_no].get("retrieved_context")
            
            prompt = f"Question: {query} \n\nRelevant documents: "
            for i in range(k):
                prompt += (retrieved_content[i] + "\n")
            prompt += "\nAnswer: "

            torch.cuda.synchronize()
            start_time = time.time()
            generation_reranker = generate_with_loop(
                prompt,
                ""
            )
            

            answer_reranker = ""
            for ans in generation_reranker:
                answer_reranker = ans
            torch.cuda.synchronize()
            end_time = time.time()
            
            generated_answers.append(answer_reranker)
            generation_time.append((end_time - start_time) * 1000)
            '''response_deepseek = query_ollama(prompt)
            generated_answers_deepseek.append(extract_after_last_think(response_deepseek))'''
            
        #generated_answer_file = os.path.join(result_dir, config['generated_answer_file'])
        generated_answer_file = os.path.join(result_dir, config['result_file'])
        #with open(generated_answer_file, "w") as answer_file:
        for query_no, answer in enumerate(generated_answers):
                #answer_file.write(f"Answer {query_no} :\n{answer}\n\n")
            retrieval_data[query_no][f"answer_{k}"] = answer
            retrieval_data[query_no][f"generation_time_{k}"] = generation_time[query_no]
            
        '''for query_no, answer in enumerate(generated_answers_deepseek):
                #answer_file.write(f"Answer {query_no} :\n{answer}\n\n")
            retrieval_data[query_no][f"answer_{k}_deepseek"] = answer'''
        
        with open(generated_answer_file, "w") as retrieval_file:
            json.dump(retrieval_data, retrieval_file, indent=4)

        print(f"Generation for {k} completed.")        


if __name__ == "__main__":   
    main(config)
    
