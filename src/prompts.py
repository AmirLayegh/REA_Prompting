EXTRACT_ENTITY_TYPE_PROMPT = """Given the below 'input text', 'head entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Country, State, Person, Organization, City, Date, Religion, Crime, Website, Person's title, University/School, political/religious_affiliation.
Example Input Text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example Head Entity: Steve Jobs
Example Tail Entity: San Francisco
Example Entity Types: 
Steve Jobs is a Person, and San Francisco is a City.


Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}"""

REFINEMENT_LABELS_PROMPT = """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.
Example input text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example head entity: Steve Jobs
Example tail entity: San Francisco
Example entity types:
Steve Jobs is a Person, and San Francisco is a City
Example list of pre-defined relation labels: {relation_labels}
Example Refined relation labels:
1. per:city_of_birth (most relevant)
2. per:city_of_death (second most relevant)
3. per:cities_of_residence (third most relevant)

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

LABELS_MAPPING_PROMPT = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', if the refined relation labels are not in the pre-defined list of relation labels, map the refined relation labels to the most relevant pre-defined relation label.
Example input text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example head entity: Steve Jobs
Example tail entity: San Francisco
Example of pre-defined list of relation labels: {relation_labels}
Example Refined relation labels:
1. per:place_of_birth (most relevant)
2. per:place_of_death (second most relevant)
3. per:cities_of_residence (third most relevant)
Example Mapped relation labels:
1. per:city_of_birth (most relevant)
2. per:city_of_death (second most relevant)
3. per:cities_of_residence (third most relevant)

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual pre-defined list of relation labels: {relation_labels}
Actual Refined Relation Labels: {refined_relation_labels}"""


CONFIDENCE_SCORE_PROMPT = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', determine the relationship confidence score for each of the three refined relation labels in the form of:

(Head Entity) (Head entity type) has been [RELATIONSHIP] (Tail Entity) (Tail entity type) with a confidence level of [CONFIDENCE].
ONLY USE THE HEAD ENTITY AND TAIL ENTITY MENTIONED IN THE INPUT TEXT WHEN DETERMINING THE RELATIONSHIP CONFIDENCE SCORE. DO NOT USE ANY OTHER INFORMATION.

Example input text: Koch Foods Inc. one of the largest poultry processors in the U.S., founded in 1973 by John Koch, in Chicago, Illinois.
Example head entity: Koch Foods Inc.
Example tail entity: Chicago
Example refined relation labels:
1. org:city_of_headquarters (most relevant)
2. org:top_members/employees (second most relevant)
3. per:employee_of (third most relevant)
Example relationship confidence scores:
(Koch Foods Inc.) organization has been headquartered in (Chicago) city with a confidence level of 100%.
(Koch Foods Inc.) organization has employed (Chicago) city with a confidence level of 0%.
(Koch Foods Inc.) person has been employed in (Chicago) city with a confidence level of 0%.
Example Explanation:
The first relationship confidence score is high because the head entity is an organization and the tail entity is a city in the input text. Also the fact of the confidence score sentence is true.
The second and third relationship confidence scores are low because the head entity is not a person in the input text.

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

refined relation labels: {refined_relation_labels}"""

RELATION_EXTRACTION_PROMPT = """Given the 'input text', 'head entity', 'tail entity', 'refined relation labels', and 'relationship confidence scores', determine the most appropriate relationship between the head entity and tail entity.
Example: the most appropriate relationship is: org:city_of_headquarters.
If there is no appropriate relationship, select 'No Relation'.

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

refined relation labels: {refined_relation_labels}

relationship confidence scores: {relationship_confidence_scores}"""

