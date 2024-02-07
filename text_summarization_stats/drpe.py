from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


parser = JsonOutputParser()


def run_DRPE(
    model,
    input_document: str,
    summary: str,
    coarse_grained_prompt_template: str,
    fine_grained_prompt_template: str,
    static_roles: List[str],
    few_shot_prompt: str,
    comparison_prompt_template: str,
    suffix_prompt_template: str,
    embedding_gnerator: str = 'all-MiniLM-L6-v2',
    roles_clusters: int = 4,
) -> float:
    # We use the dynamic_roles_generator function which inserts the input document in the prompt template and runs the model.
    # The model must be a class with a __call__ function that calls the model and extracts the output correctly.
    coarse_grained_roles, fine_grained_roles = dynamic_roles_generator(model, input_document, coarse_grained_prompt_template, fine_grained_prompt_template)

    # Extract the roles from text with the dynamic roles parser
    roles = dynamic_role_parser(coarse_grained_roles) + dynamic_role_parser(fine_grained_roles)

    # Cluster the roles with the dynamic roles clusteres which uses the sentencetransformer package and kmeans.
    roles_clustered = dynamic_roles_clutering(roles, embedding_gnerator, roles_clusters)

    # Add the static roles to the clusterd roles
    roles_clustered.extend(static_roles)

    # Run the evaluation
    evaluation = evaluator(
        model,  # the LLM class with a __call__ function
        input_document,  # the input document
        summary,  # the summary that must be evaluated
        roles_clustered,  # the clustered roles
        few_shot_prompt,  # the few shot prompt template is used to insert the generated roles in the prompt
        comparison_prompt_template,  # beginning of the prompt
        suffix_prompt_template,  # this is added at the end. The evaluator gives an error without it
    )

    # Parse the output
    try:
        parsed_evaluation = parser.parse(evaluation)
    except:
        parsed_evaluation = []

    if not parsed_evaluation:
        scores = []
        counts = {}
    else:
        scores = [x['preferred_summary'] for x in parsed_evaluation]
        counts = Counter(scores)

    return scores, counts


def dynamic_roles_generator(
    model,
    input_text: str,
    coarse_grained_prompt_template: str,
    fine_grained_prompt_template: str,
) -> Tuple[str, str]:
    """ Generates the dynamic roles

    Args:
        model: The model used to generate the dynamic roles.
        input_text: The input text.

    Returns:
        A tuple containing the coarse grained roles and the fine grained roles.
        The tuple has two elements since the output is unpocessed text from the LLM.
    """
    # Generate coarse grained roles
    coarse_grained_prompt = PromptTemplate.from_template(coarse_grained_prompt_template)
    model.add_prompt(coarse_grained_prompt)
    coarse_grained_roles = model(input={'text': input_text})

    # Generate fine grained roles
    fine_grained_prompt = PromptTemplate.from_template(fine_grained_prompt_template)
    model.add_prompt(fine_grained_prompt)
    fine_grained_roles = model(input={'text': input_text})

    return coarse_grained_roles, fine_grained_roles


def dynamic_role_parser(
    text: str
) -> Dict[str, str]:
    """ Parses the output of LLM to extract the dynamic roles

    Args:
        text: The output of LLM.

    Returns:
        A list of roles.

    Warning:
        This function is not robust to changes in the output of LLM.
        For as it is right now the LLM is expected to follow the following format:
            1. role 1: role description
            2. role 2: role description
            ...
    """
    # parse the roles
    roles_raw = [x.strip() for x in text.split('\n')]
    roles_raw = list(filter(lambda x: x, roles_raw))
    roles = [x[3:].strip() for x in roles_raw]

    # Save the role types and descriptions
    types_descriptions = []
    for role in roles:
        role_split = role.split(':')
        types_descriptions.append([
            role_split[0].strip(),
            ':'.join(role_split[1:]).strip()
        ])

    return types_descriptions


def dynamic_roles_clutering(
    generated_roles: List[List[str]],
    model_name: str,
    n_clusters: int
):
    """ Clusters text using embeddings and k-means """
    # Create the embeddings
    embedding_model = SentenceTransformer(model_name)
    roles_concatenated = [': '.join(x) for x in generated_roles]
    if 'uncased' in model_name.lower():
        sentence_embeddings = embedding_model.encode([el.lower() for el in roles_concatenated])
    else:
        sentence_embeddings = embedding_model.encode(roles_concatenated)

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(sentence_embeddings)

    # Find the elements closest to the centroids
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    roles = []
    for i in range(n_clusters):
        cluster_elements = list(filter(lambda x: x[2] == i, zip(generated_roles, sentence_embeddings, labels)))
        distances = [distance_measure(centers[i], el[1]) for el in cluster_elements]
        closest_index = np.argmin(distances)
        roles.append(cluster_elements[closest_index][0])

    return roles


def distance_measure(
    vector_a,
    vector_b
):
    return sum([(a - b) ** 2 for a, b in zip(vector_a, vector_b)])


def evaluator(
    model,
    input_document: str,
    summaries: List[str],
    roles: List[List[str]],
    few_shot_prompt: str,
    comparison_prompt_template: str,
    suffix_prompt_template: str,
):
    """ Runs the evaluation of two summaries using the dynamic roles """
    # Define the formatted roles for the few shot prompt template
    examples = []
    for line in roles:
        if len(line) < 2:
            continue
        examples.append({
            'role_type': line[0],
            'role_description': line[1],
        })

    example_prompt = PromptTemplate(
        input_variables=["role_type", "role_description"], template=few_shot_prompt
    )
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=comparison_prompt_template,
        input_variables=['text', 'summary_1', 'summary_2'],
        suffix=suffix_prompt_template,
    )

    # Evaluate the summaries
    model.add_prompt(prompt)

    return model(input={
        'text': input_document.replace('\n', ' ').replace('\t', ' ').replace('  ', ' '),
        'summary_1': summaries[0],
        'summary_2': summaries[1]
    })
