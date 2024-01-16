EXTRACT_ENTITY_TYPE_PROMPT = """Given the below 'input text', 'head entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Country, State, Person, Organization, City, Date, Religion, Crime, Website, Person's title, University/School, Political Affiliation, Religious Affiliation, Nationality/Origin.
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
Example Explanation:
The first relation label is the most relevant because the head entity is a person and the tail entity is a city in the input text. Also it is mentioned in the input text that Steve Jobs was born in San Francisco.
The second and third relation labels are also relevant because the head entity is a person and the tail entity is a city in the input text. However, the second relation label is more relevant than the third relation label because it is mentioned in the input text that Steve Jobs was born in San Francisco.

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
DO NOT ADD ANY OTHER ADDITIONAL WORDS SUCH AS ADVERBS OR ADJECTIVES TO THE RELATIONSHIP CONFIDENCE SCORE SENTENCE.

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

CONFIDENCE_SCORE_PROMPT2= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:
Example input text: John Smith an English scientist in the fielad of astronomy, invented the reflecting telescope in 1668.
Example head entity: John Smith
Example tail entity: English
Example entity types:
Johns Smith is a person and English is a nationality.
Example refined relation labels:
1. per:origin (most relevant)
2. per:political/religious_affiliation (second most relevant)
3. per:title (third most relevant)
Example relationship confidence scores sentences:
1. Person John Smith has been originated from English with a confidence level of 100%.
2. Person John Smith has been politically/religiously affiliated with English with a confidence level of 0%.
3. Person John Smith has been titled as English with a confidence level of 0%.

Example Explanation:
The confidence score for the first sentence is high because the head entity type is a person and the tail entity type is nationality. Also the fact of the confidence score sentence is true.
The confidence scores for the second sentence is low because the tail entity type is not a political/religious affiliation in the input text. Also the fact of the confidence score sentence is false.
The confidence scores for the third sentence is low because the tail entity type is not a title in the input text. Also the fact of the confidence score sentence is false.

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

entity types: {entity_types}

refined relation labels: {refined_relation_labels}"""

RELATION_EXTRACTION_PROMPT = """Given the 'input text', 'head entity', 'tail entity', 'refined relation labels', and 'relationship confidence scores', determine the most appropriate relationship between the head entity and tail entity.
Example: the most appropriate relationship is: org:city_of_headquarters.

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

refined relation labels: {refined_relation_labels}

relationship confidence scores: {relationship_confidence_scores}"""