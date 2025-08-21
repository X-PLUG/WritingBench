import os
import json
import glob
import pandas as pd
from collections import defaultdict


DOMAIN1_ORDER = [
    'Academic & Engineering',
    'Finance & Business', 
    'Politics & Law',
    'Literature & Arts',
    'Education',
    'Advertising & Marketing'
]

REQUIREMENT_DIMENSION = ["style", "format", "length"]


def read_scores_file(jsonl_file_path):
    scores_data = {}
    scores_data_details = {}
    total_scores_sum = defaultdict(lambda: 0)
    total_count = defaultdict(lambda: 0)
    query_count = 0

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            index = record['index']
            scores = record['scores']
            scores_data_details[index] = {}
            avg_score = 0
            total_criteria = 0
            for criteria, evaluations in scores.items():
                criteria_lis = []
                for eval in evaluations:
                    avg_score += eval['score']
                    total_scores_sum[index] += eval['score']
                    total_count[index] += 1
                    total_criteria += 1
                    criteria_lis.append(eval['score'])
                scores_data_details[index][criteria] = sum(criteria_lis) / len(criteria_lis)

            avg_score = avg_score / total_criteria
            scores_data[index] = avg_score
            query_count += 1

    overall_avg = sum(total_scores_sum.values()) / sum(total_count.values())
    return scores_data, overall_avg, query_count, scores_data_details


def read_requirement_file(requirement_dir):
    requirement_R = {}
    requirement_C = {}
    for requirement in REQUIREMENT_DIMENSION:
        requirement_R[requirement] = []
        requirement_C[requirement] = {}
        file_path_R = os.path.join(requirement_dir, requirement, f"{requirement}_subset.jsonl")
        file_path_C = os.path.join(requirement_dir, requirement, f"{requirement}_subset_C.jsonl")

        with open(file_path_R, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                index = record['index']
                requirement_R[requirement].append(index)

        with open(file_path_C, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                index = record['index']
                requirement_C[requirement][index] = []
                for criteria in record["checklist"]:
                    requirement_C[requirement][index].append(criteria["name"])
    
    return requirement_R, requirement_C


def read_domain_file(jsonl_file_path):
    domain_data = {}
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            index = record['index']
            domain1 = record.get('domain1', None)
            domain2 = record.get('domain2', None)
            domain_data[index] = {'domain1': domain1, 'domain2': domain2}
    return domain_data


def calculate_domain_scores(scores_data, domain_data):
    domain1_scores_sum = defaultdict(lambda: 0)
    domain1_count = defaultdict(lambda: 0)
    
    domain2_scores_sum = defaultdict(lambda: 0)
    domain2_count = defaultdict(lambda: 0)

    for index, score in scores_data.items():
        if index in domain_data:
            domains = domain_data[index]
            domain1 = domains['domain1']
            domain2 = domains['domain2']

            if domain1 is not None:
                domain1_scores_sum[domain1] += score
                domain1_count[domain1] += 1

            if domain2 is not None:
                domain2_scores_sum[domain2] += score
                domain2_count[domain2] += 1

    domain1_avg_scores = {domain: domain1_scores_sum[domain] / domain1_count[domain]
                          for domain in domain1_scores_sum}
    
    domain2_avg_scores = {domain: domain2_scores_sum[domain] / domain2_count[domain]
                          for domain in domain2_scores_sum}

    return domain1_avg_scores, domain2_avg_scores


def calculate_requirement_scores(scores_data, scores_data_details, requirement_R, requirement_C):
    requirement_R_score = {}
    requirement_C_score = {}
    for requirement in REQUIREMENT_DIMENSION:
        score_R = []
        score_C = []
        for index in requirement_R[requirement]:
            if index not in scores_data:
                continue
            score_R.append(scores_data[index])

            for name in requirement_C[requirement][index]:                
                score_C.append(scores_data_details[index][name])
        requirement_R_score[requirement] = sum(score_R) / len(score_R) if len(score_R) else 0
        requirement_C_score[requirement] = sum(score_C) / len(score_C) if len(score_C) else 0
    return requirement_R_score, requirement_C_score


def aggregate_scores(input_directory, domain_file, output_excel_file, requirement_dir):
    requirement_R, requirement_C = read_requirement_file(requirement_dir)
    domain_data = read_domain_file(domain_file)
    domain2_to_domain1 = {}
    for idx, domains in domain_data.items():
        if domains['domain2']:
            domain2_to_domain1[domains['domain2']] = domains['domain1']

    jsonl_files = glob.glob(os.path.join(input_directory, '*.jsonl'))
    per_model_results: Dict[str, Dict[str, Any]] = {}
    all_domain2_scores: DefaultDict[str, List[float]] = defaultdict(list)
    for jsonl_file in jsonl_files:
        name = os.path.basename(jsonl_file)
        if name.endswith('.jsonl'):
            name = name[:-len('.jsonl')]
        scores_data, overall_avg, query_count, scores_data_details = read_scores_file(jsonl_file)
        domain1_avg_scores, domain2_avg_scores = calculate_domain_scores(scores_data, domain_data)
        results: Dict[str, Any] = {"Model": name, "Overall": overall_avg, "Count": query_count}
        for domain, avg_score in domain1_avg_scores.items():
            results[f'Domain1_{domain}'] = avg_score
    
        for domain, avg_score in domain2_avg_scores.items():
            results[f'Domain2_{domain}'] = avg_score
            all_domain2_scores[domain].append(avg_score)
        requirement_R_score, requirement_C_score = calculate_requirement_scores(scores_data, scores_data_details, requirement_R, requirement_C)
        for requirement in REQUIREMENT_DIMENSION:
            results[f"{requirement}_R"] = requirement_R_score[requirement]
            results[f"{requirement}_C"] = requirement_C_score[requirement]
        
        per_model_results[name] = results

    domain_hierarchy = {d1: [] for d1 in DOMAIN1_ORDER}
    for domain2, scores in all_domain2_scores.items():
        if domain2 in domain2_to_domain1:
            d1 = domain2_to_domain1[domain2]
            avg_score = sum(scores) / len(scores)
            domain_hierarchy[d1].append((domain2, avg_score))

    sorted_domains = {}
    for d1 in DOMAIN1_ORDER:
        sorted_d2 = sorted(domain_hierarchy[d1], 
                          key=lambda x: x[1], 
                          reverse=True)
        sorted_domains[d1] = [d2[0] for d2 in sorted_d2]
    
    rows = []
    for model, result in per_model_results.items():
        ordered = {"Model": result["Model"], "Overall": result["Overall"], "Count": result["Count"]}
        for d1 in DOMAIN1_ORDER:
            ordered[f"Domain1_{d1}"] = result.get(f"Domain1_{d1}")
        for d1 in DOMAIN1_ORDER:
            for d2 in sorted_domains.get(d1, []):
                ordered[f"Domain2_{d2}"] = result.get(f"Domain2_{d2}")
        for req in REQUIREMENT_DIMENSION:
            ordered[f"{req}_R"] = result.get(f"{req}_R")
            ordered[f"{req}_C"] = result.get(f"{req}_C")
        rows.append(ordered)

    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Aggregate score files, compare with the benchmark, and export results to an Excel spreadsheet.")
    parser.add_argument("--score_dir", type=str, required=True, help="Directory that contains per-sample score JSONL files.")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to the benchmark file (JSONL) used for reference.")
    parser.add_argument("--output_excel", type=str, required=True, help="Path of the Excel file to write aggregated results.")
    parser.add_argument("--requirement_dir", type=str, required=True, help="Directory that stores requirement files.")
    args = parser.parse_args()

    aggregate_scores(args.score_dir, args.benchmark_file, args.output_excel, args.requirement_dir)

