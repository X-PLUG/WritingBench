import json
import os
import argparse
import jsonlines
from tqdm import tqdm
from prompt import evaluate_system, evaluate_prompt
from evaluator import ClaudeAgent, CriticAgent

EVAL_TIMES = 1

class EvalAgent(object):
    def __init__(self, agent):
        self.agent = agent
    
    def success_check_fn_score(self, response):
        try:
            result = json.loads(response.strip('json|```'))
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return False
        
        valid_score_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        if "score" not in result or "reason" not in result:
            print("Missing 'score' or 'reason' in the result")
            return False
        if result["score"] not in valid_score_values:
            return False
        if not isinstance(result["reason"], str):
            return False
        return True


    def generate_score(self, content, query, criteria):
        prompt_data = {
            "query": query,
            "response": content["response"],
            "criteria": criteria,
        }
        retry = 0
        success = False
        while not success and retry < 3:
            prompt = evaluate_prompt.format(**prompt_data)
            response, success = self.agent.run(
                prompt=prompt,
                success_check_fn=self.success_check_fn_score
            )
            try:
                response = json.loads(response.strip('json|```'))
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                response = eval(response.strip('json|```'))
            retry += 1
        if success:
            return response
        else:
            raise ValueError("Fail to generate score!")

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

def load_query_criteria(jsonl_file_path):
    """
    Loads criteria from a JSONL file into a dictionary.
    """
    data_list = {}
    with jsonlines.open(jsonl_file_path) as reader:
        for obj in reader:
            data_list[obj['index']] = {}
            data_list[obj['index']]['query'] = obj['query']
            data_list[obj['index']]['criteria'] = obj['checklist']
    return data_list

def process(agent, input_file, out_file, id_query_criteria_map):
    """
    Processes input files through the evaluation agent, producing scores and saving results.
    """
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(input_file)
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {input_file.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            if existing_count > 0 and i < existing_count - 1:
                pbar.update()
                continue
            
            data = {
                "index": content["index"],
                "scores": {}
            }

            query = id_query_criteria_map[content["index"]]['query']
            criteria = id_query_criteria_map[content["index"]]['criteria']

            with tqdm(total=len(criteria) * EVAL_TIMES, desc=f"Data ID {content['index']} Progress", leave=False) as internal_pbar:
                for c in criteria:
                    if c["name"] not in criteria:
                        data["scores"][c["name"]] = []
                    while len(data["scores"][c["name"]]) < EVAL_TIMES:
                        score = agent.generate_score(content, query, c)
                        data["scores"][c["name"]].append(score)
                        internal_pbar.update(1)

            save_output([data], out_file)
            cnt += 1
            pbar.update()

        print(f"CNT: {cnt}")

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--evaluator", choices=['claude', 'critic'], required=True, help="Choose the scoring model to use: 'claude' or 'critic'.")
    parser.add_argument("--query_criteria_file", type=str, help="Path to the query and criteria file.")
    parser.add_argument("--input_file", type=str, help="Path to the input file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")

    args = parser.parse_args()

    # Evaluator initialization based on chosen model
    if args.evaluator == 'claude':
        agent = EvalAgent(ClaudeAgent(
            system_prompt=evaluate_system,
        ))
    else:
        agent = EvalAgent(CriticAgent(
            system_prompt=evaluate_system,
        ))

    id_query_criteria_map = load_query_criteria(args.query_criteria_file)

    process(agent, args.input_file, args.output_file, id_query_criteria_map)
