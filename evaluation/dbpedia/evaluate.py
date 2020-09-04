"""Evaluation script for the SMART Task.

Usage:
    `python evaluate.py type_hierarchy_tsv ground_truth_json system_output_json`

Where
    - `type_hierarchy_tsv` is a TSV file with Type, Depth and Parent columns.
      The file is assumed to contain a header row.
    - `ground_truth_json` is a JSON file containing the input questions and the
      ground truth category and list of types (following the format of the
      training data files).
    - `system_output_json` is a JSON file with the (participating) system's
      category and type predictions. The format is a list of dictionaries with
      keys `id`, `category`, and `type`, holding the question ID, predicted
      category, and ranked list of up to 10 types, respectively.


The script computes accuracy for category classification and NDCG@k for type
prediction.

Specifically, type prediction is computed only for questions that fall into the
'literal' or 'resource' answer categories.
    - For 'literal' answer types, only a single predicted type is considered
      that can be either correct (NDCG=1) or incorrect (NDCG=0).
    - For 'resource' answer types a ranked list of 10 ontology classes is
      considered and evaluated in terms of lenient NDCG@k with linear decay
      (according to (Balog and Neumayer, CIKM'12)).

Author: Krisztian Balog (krisztian.balog@uis.no)
"""

import argparse
import json
import math


def load_type_hierarchy(filename):
    """Reads the type hierarchy from a TSV file.

    Note: The TSV file is assumed to have a header row.

    Args:
        filename: Name of TSV file.

    Returns:
        A tuple with a types dict and the max hierarchy depth.
    """
    print('Loading type hierarchy from {}... '.format(filename), end='')
    types = {}
    max_depth = 0
    with open(filename, 'r') as tsv_file:
        next(tsv_file)  # Skip header row
        for line in tsv_file:
            fields = line.rstrip().split('\t')
            type_name, depth, parent_type = fields[0], int(fields[1]), fields[2]
            types[type_name] = {'parent': parent_type,
                                'depth': depth}
            max_depth = max(depth, max_depth)
    print('{} types loaded (max depth: {})'.format(len(types), max_depth))
    return types, max_depth


def load_ground_truth(filename, type_hierarchy):
    """Loads the ground truth from a JSON file."""
    print('Loading ground truth from {}... '.format(filename))
    ground_truth = {}
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        for question in data:
            if not question['question']:  # Ignoring null questions
                print('WARNING: question text for ID {} is empty'.format(
                    question['id']))
                continue
            types = []
            for type in question['type']:
                if question['category'] == 'resource' \
                        and type not in type_hierarchy:
                    print('WARNING: unknown type "{}"'.format(type))
                    continue
                types.append(type)
            ground_truth[question['id']] = {
                'category': question['category'],
                'type': types
            }
    print('   {} questions loaded'.format(len(ground_truth)))
    return ground_truth


def load_system_output(filename):
    """Loads the system's predicted output from a JSON file."""
    print('Loading system predictions from {}... '.format(filename))
    system_output = {}
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        for answer in data:
            system_output[answer['id']] = {
                'category': answer['category'],
                'type': answer['type']
            }
    print('   {} predictions loaded'.format(len(system_output)))
    return system_output


def dcg(gains, k=5):
    """Computes DCG for a given ranking.

    Traditional DCG formula: DCG_k = sum_{i=1}^k gain_i / log_2(i+1).
    """
    dcg = 0
    for i in range(0, min(k, len(gains))):
        dcg += gains[i] / math.log(i + 2, 2)
    return dcg


def ndcg(gains, ideal_gains, k=5):
    """Computes NDCG given gains for a ranking as well as the ideal gains."""
    return dcg(gains, k) / dcg(ideal_gains, k)


def get_type_path(type, type_hierarchy):
    """Gets the type's path in the hierarchy (excluding the root type, like
    owl:Thing).

    The path for each type is computed only once then cached in type_hierarchy,
    to save computation.
    """
    if 'path' not in type_hierarchy[type]:
        type_path = []
        current_type = type
        while current_type in type_hierarchy:
            type_path.append(current_type)
            current_type = type_hierarchy[current_type]['parent']
        type_hierarchy[type]['path'] = type_path
    return type_hierarchy[type]['path']


def get_type_distance(type1, type2, type_hierarchy):
    """Computes the distance between two types in the hierarchy.

    Distance is defined to be the number of steps between them in the hierarchy,
    if they lie on the same path (which is 0 if the two types match), and
    infinity otherwise.
    """
    type1_path = get_type_path(type1, type_hierarchy)
    type2_path = get_type_path(type2, type_hierarchy)
    distance = math.inf
    if type1 in type2_path:
        distance = type2_path.index(type1)
    if type2 in type1_path:
        distance = min(type1_path.index(type2), distance)
    return distance


def get_most_specific_types(types, type_hierarchy):
    """Filters a set of input types to most specific types w.r.t the type
    hierarchy; i.e., super-types are removed."""
    filtered_types = set(types)
    for type in types:
        type_path = get_type_path(type, type_hierarchy)
        for supertype in type_path[1:]:
            if supertype in filtered_types:
                filtered_types.remove(supertype)
    return filtered_types


def get_expanded_types(types, type_hierarchy):
    """Expands a set of types with both more specific and more generic types
    (i.e., all super-types and sub-types)."""
    expanded_types = set()
    for type in types:
        # Adding all supertypes.
        expanded_types.update(get_type_path(type, type_hierarchy))
        # Adding all subtypes (NOTE: this bit could be done more efficiently).
        for type2 in type_hierarchy:
            if type_hierarchy[type2]['depth'] <= type_hierarchy[type]['depth']:
                continue
            type2_path = get_type_path(type2, type_hierarchy)
            if type in type2_path:
                expanded_types.update(type2_path)
    return expanded_types


def compute_type_gains(predicted_types, gold_types, type_hierarchy, max_depth):
    """Computes gains for a ranked list of type predictions.

    Following the definition of Linear gain in (Balog and Neumayer, CIKM'12),
    the gain for a given predicted type is 0 if it is not on the same path with
    any of the gold types, and otherwise it's $1-d(t,t_q)/h$ where $d(t,t_q)$ is
    the distance between the predicted type and the closest matching gold type
    in the type hierarchy and h is the maximum depth of the type hierarchy.

    Args:
        predicted_types: Ranked list of predicted types.
        gold_types: List/set of gold types (i.e., perfect answers).
        type_hierarchy: Dict with type hierarchy.
        max_depth: Maximum depth of the type hierarchy.

    Returns:
        List with gain values corresponding to each item in predicted_types.
    """
    gains = []
    expanded_gold_types = get_expanded_types(gold_types, type_hierarchy)
    for predicted_type in predicted_types:
        if predicted_type in expanded_gold_types:
            # Since not all gold types may lie on the same branch, we take the
            # closest gold type for determining distance.
            min_distance = math.inf
            for gold_type in gold_types:
                min_distance = min(get_type_distance(predicted_type, gold_type,
                                                     type_hierarchy),
                                   min_distance)
            gains.append(1 - min_distance / max_depth)
        else:
            gains.append(0)
    return gains


def evaluate(system_output, ground_truth, type_hierarchy, max_depth):
    """Evaluates a system's predicted output against the ground truth.

    Args:
        system_output: Dict with the system's predictions.
        ground_truth: Dict with the ground truth.
        type_hierarchy: Dict with the type hierarchy.
        max_depth: Maximum depth of the type hierarchy.
    """
    accuracy = []
    ndcg_5, ndcg_10 = [], []
    for question_id, gold in ground_truth.items():
        if question_id not in system_output:
            print('WARNING: no prediction made for question ID {}'.format(
                question_id))
            system_output[question_id] = {}
        predicted_category = system_output[question_id].get('category', None)
        predicted_type = system_output[question_id].get('type', [None])

        if predicted_category != gold['category']:
            accuracy.append(0)
            continue

        # Category has been correctly predicted -- proceed to type evaluation.
        accuracy.append(1)

        if gold['category'] == 'literal':
            gains = [1 if gold['type'][0] == predicted_type[0] else 0]
            ideal_gains = [1]
        elif gold['category'] == 'resource':
            if len(gold['type']) == 0:
                print('WARNING: no gold types given for question ID {}'.format(
                    question_id))
                continue
            # Filters gold types to most specific ones in the hierarchy.
            gold_types = get_most_specific_types(gold['type'], type_hierarchy)

            gains = compute_type_gains(predicted_type, gold_types,
                                       type_hierarchy, max_depth)
            ideal_gains = sorted(
                compute_type_gains(
                    get_expanded_types(gold_types, type_hierarchy), gold_types,
                    type_hierarchy, max_depth), reverse=True)
        else:
            continue

        ndcg_5.append(ndcg(gains, ideal_gains, k=5))
        ndcg_10.append(ndcg(gains, ideal_gains, k=10))

    print('\n')
    print('Evaluation results:')
    print('-------------------')
    print('Category prediction (based on {} questions)'.format(
        len(accuracy)))
    print('  Accuracy: {:5.3f}'.format(sum(accuracy) / len(accuracy)))
    print('Type ranking (based on {} questions)'.format(len(ndcg_5)))
    print('  NDCG@5:  {:5.3f}'.format(sum(ndcg_5) / len(ndcg_5)))
    print('  NDCG@10: {:5.3f}'.format(sum(ndcg_10) / len(ndcg_10)))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('type_hierarchy_tsv', type=str,
                        help='type hierarchy TSV file')
    parser.add_argument('ground_truth_json', type=str,
                        help='ground truth JSON file')
    parser.add_argument('system_output_json', type=str,
                        help='system output JSON file')
    args = parser.parse_args()
    return args


def main(args):
    type_hierarchy, max_depth = load_type_hierarchy(args.type_hierarchy_tsv)
    ground_truth = load_ground_truth(args.ground_truth_json, type_hierarchy)
    system_output = load_system_output(args.system_output_json)
    evaluate(system_output, ground_truth, type_hierarchy, max_depth)


if __name__ == "__main__":
    main(arg_parser())
