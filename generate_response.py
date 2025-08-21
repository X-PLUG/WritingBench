import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI


def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

# === Your code here ===

# def writer(query):
#     try:
#         client = OpenAI(
#             api_key="API KEY",
#             base_url="BASE_URL",
#         )

#         completion = client.chat.completions.create(
#             model="model_name",
#             messages=[
#                 {"role": "user", "content": query},
#             ],
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         print(f"error: {e}")
#         raise
    
def process(id_query_map, out_file):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            if existing_count > 0 and i < existing_count: 
                pbar.update()
                continue
            data = {"index": content["index"]}
            query = content["query"]
            data["response"] = writer(query)
            save_output([data], out_file)
            cnt += 1
            pbar.update()

    print(f"CNT: {cnt}")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--query_file", type=str, help="Path to the query file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    parser.add_argument("--model", type=str, help="Path to the output file.")

    args = parser.parse_args()

    process(args.query_file, args.output_file)

