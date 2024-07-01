EXTRACT_ENTITY_TYPE_PROMPT = """Given the below 'input text', 'head entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Country, State, Person, Organization, City, Date, Religion, Crime, Website, Person's title, University/School, Political Affiliation, Religious Affiliation, Nationality/Origin, Festival.
Example Input Text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example Head Entity: Steve Jobs
Example Tail Entity: San Francisco
Example Entity Types: 
Steve Jobs is a Person, and San Francisco is a City.


Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}"""

# Zero-shot version
EXTRACT_ENTITY_TYPE_PROMPT_ZERO_SHOT = """Given the below 'input text', 'head entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Country, State, Person, Organization, City, Date, Religion, Crime, Website, Person's title, University/School, Political Affiliation, Religious Affiliation, Nationality/Origin, Festival.

Input Text: {sentence}
Head Entity: {head_entity}
Tail Entity: {tail_entity}

Please provide the entity types for the head and tail entities."""

EXTRACT_ENTITY_TYPE_PROMPT_WIKI= """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.
Example Input Text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example Head Entity: Steve Jobs
Example Tail Entity: San Francisco
Example Entity Types:
Steve Jobs is a Person, and San Francisco is a Location.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}"""

# Zero-shot version
EXTRACT_ENTITY_TYPE_PROMPT_WIKI_ZERO_SHOT = """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.

Input Text: {sentence}
Head Entity: {head_entity}
Tail Entity: {tail_entity}

Please provide the entity types for the head and tail entities."""

REFINEMENT_LABELS_PROMPT_WIKI = """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.
Example input text: Steve Jobs lived in San Francisco, for 10 years.
Example head entity: Steve Jobs
Example tail entity: San Francisco
Example entity types:
Steve Jobs is a Person, and San Francisco is a Location
Example list of pre-defined relation labels: ["residence", "twinned administrative body", "cast member", "league", "located on astronomical body"]
Example Refined relation labels:
1. residence (most relevant)
2. league (second most relevant)
3. cast member (third most relevant)

Example Explanation:
The first relation label is the most relevant because the head entity is a person and the tail entity is a location in the input text. Also it is mentioned in the input text that Steve Jobs lived in San Francisco.
The second relation label is less relevant because the head entity is a person and the tail entity is a location in the input text. Also it is not mentioned in the input text that Steve Jobs was a member of a league.
The third relation label is the least relevant because the head entity is a person and the tail entity is a location in the input text. Also it is not mentioned in the input text that Steve Jobs was a cast member.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

# Zero-shot version
REFINEMENT_LABELS_PROMPT_WIKI_ZERO_SHOT = """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_WIKI_PROMPT = """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.
After categorizing the entity types, analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.
Example Input Text: Steve Jobs was born in San Francisco, on February 24, 1955.
Example Head Entity: Steve Jobs
Example Tail Entity: San Francisco
Example Entity Types:
Steve Jobs is a Person, and San Francisco is a City
Example list of pre-defined relation labels: {relation_labels}
Example Refined relation labels:
1. residence (most relevant)
2. league (second most relevant)
3. cast member (third most relevant)

Example Explanation:
The first relation label is the most relevant because the head entity is a person and the tail entity is a city in the input text. Also it is mentioned in the input text that Steve Jobs lived in San Francisco.
The second relation label is less relevant because the head entity is a person and the tail entity is a location in the input text. Also it is not mentioned in the input text that Steve Jobs was a member of a league.
The third relation label is the least relevant because the head entity is a person and the tail entity is a location in the input text. Also it is not mentioned in the input text that Steve Jobs was a cast member.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual List of Pre-defined Relation Labels: {relation_labels}."""

# Zero-shot version
EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_WIKI_PROMPT_ZERO_SHOT = """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.
After categorizing the entity types, analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual List of Pre-defined Relation Labels: {relation_labels}."""

EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_FEWREL_PROMPT = """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.
After categorizing the entity types, analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.
Example Input Text: Sarah Jessica Parker mesmerized the audience as the lead dancer, in the Holloween party.
Example Head Entity: Sarah Jessica Parker
Example Tail Entity: Holloween
Example Entity Types:
Sarah Jessica Parker is a Person, and Holloween is a Festival
Example list of pre-defined relation labels: {relation_labels}
Example Refined relation labels:
1. performer (most relevant)
2. work location (second most relevant)
3. genre (third most relevant)

Example Explanation:
The first relation label is the most relevant because the head entity is a person and the tail entity is a festival in the input text. Also it is mentioned in the input text that Sarah Jessica Parker performed in the Holloween party.
The second relation label is less relevant because the head entity is a person and the tail entity is a festival in the input text. Also it is not mentioned in the input text that Sarah Jessica Parker worked in the Holloween party.
The third relation label is the least relevant because the head entity is a person and the tail entity is a festival in the input text. Also it is not mentioned in the input text that Sarah Jessica Parker was genred in the Holloween party.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

# Zero-shot version
EXTRACT_ENTITY_TYPE_AND_REFINEMENT_LABELS_FEWREL_PROMPT_ZERO_SHOT = """Given the below 'input text', 'head_entity', and 'tail entity', your task is to categorize the entity type of the head and tail entities. Use the following options for categorization: Person, Organization, Location, Date, Event, Product, Work, Other.
After categorizing the entity types, analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

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

# Zero-shot version
REFINEMENT_LABELS_PROMPT_ZERO_SHOT = """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

# Zero-shot version
REFINEMENT_LABELS_PROMPT_FEWREL_ZERO_SHOT= """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

REFINEMENT_LABELS_PROMPT_FEWREL= """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.
Example input text: Sarah Jessica Parker mesmerized the audience as the lead dancer, in the Holloween party.
Example head entity: Sarah Jessica Parker
Example tail entity: Holloween
Example entity types:
Sarah Jessica Parker is a Person, and Holloween is a Festival.
Example list of pre-defined relation labels: {relation_labels}
Example Refined relation labels:
1. performer (most relevant)
2. work location (second most relevant)
3. genre (third most relevant)

Example Explanation:
The first relation label is the most relevant because the head entity is a person and the tail entity is a festival in the input text. Also it is mentioned in the input text that Sarah Jessica Parker performed in the Holloween party.
The second relation label is also relevant because the head entity is a person and the tail entity is a festival in the input text. However, the second relation label is less relevant than the first relation label because it is not mentioned in the input text that Sarah Jessica Parker worked in the Holloween party.
The third relation label is the least relevant because the head entity is a person and the tail entity is a festival in the input text. 

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""

# Zero-shot version
REFINEMENT_LABELS_PROMPT_FEWREL_ZERO_SHOT= """Given the 'input text', 'head entity', 'tail entity', and 'entity types', analyze the list of pre-defined relation labels and select the three most relevant relation between the head entity and tail entity.
Ensure all three classes are from the pre-defined list of relation labels.

Actual Input Text: {sentence}
Actual Head Entity: {head_entity}
Actual Tail Entity: {tail_entity}
Actual Entity Types: {entity_types}
Actual List of Pre-defined Relation Labels: {relation_labels}"""


CONFIDENCE_SCORE_PROMPT= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:
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

# Zero-shot version
CONFIDENCE_SCORE_PROMPT_ZERO_SHOT= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

entity types: {entity_types}

refined relation labels: {refined_relation_labels}"""

CONFIDENCE_SCORE_PROMPT_FEWREL_SEP= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:
Example input text: Sarah Jessica Parker mesmerized the audience as the lead dancer, in the Holloween party.
Example head entity: Sarah Jessica Parker
Example tail entity: Holloween
Example entity types:
Sarah Jessica Parker is a Person, and Holloween is a Festival.
Example refined relation labels:
1. performer (most relevant)
2. work location (second most relevant)
3. genre (third most relevant)
Example relationship confidence scores sentences:
1. Person Sarah Jessica Parker has been performed in Festival Holloween with a confidence level of 100%.
2. Person Sarah Jessica Parker has been worked in Festival Holloween with a confidence level of 0%.
3. Person Sarah Jessica Parker has been genred in Festival Holloween with a confidence level of 0%.

Example Explanation:
The confidence score for the first sentence is high because the head entity type is a person and the tail entity type is festival. Also the fact of the confidence score sentence is true and it is mentioned in the input text that Sarah Jessica Parker performed in the Holloween party.
The confidence scores for the second sentence is low because the tail entity type is not a work location in the input text. Also the fact of the confidence score sentence is false.
The confidence scores for the third sentence is low because the tail entity type is not a genre in the input text. Also the fact of the confidence score sentence is false.

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types: {entity_types}
refined relation labels: {refined_relation_labels}"""

# Zero-shot version
CONFIDENCE_SCORE_PROMPT_FEWREL_SEP_ZERO_SHOT= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types: {entity_types}
refined relation labels: {refined_relation_labels}"""


CONFIDENCE_SCORE_PROMPT_FEWREL_JOINT= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels.
DO NOT ADD ANY OTHER ADDITIONAL WORDS SUCH AS ADVERBS OR ADJECTIVES TO THE RELATIONSHIP CONFIDENCE SCORE SENTENCE.
Example input text: Sarah Jessica Parker mesmerized the audience as the lead dancer, in the Holloween party.
Example head entity: Sarah Jessica Parker
Example tail entity: Holloween
Example entity types:
Sarah Jessica Parker is a Person, and Holloween is a Festival.
Example refined relation labels:
1. performer (most relevant)
2. work location (second most relevant)
3. genre (third most relevant)
Example relationship confidence scores sentences:
1. Person Sarah Jessica Parker has been performed in Festival Holloween with a confidence level of 100%.
2. Person Sarah Jessica Parker has been worked in Festival Holloween with a confidence level of 0%.
3. Person Sarah Jessica Parker has been genred in Festival Holloween with a confidence level of 0%.

Example Explanation:
The confidence score for the first sentence is high because the head entity type is a person and the tail entity type is festival. Also the fact of the confidence score sentence is true and it is mentioned in the input text that Sarah Jessica Parker performed in the Holloween party.
The confidence scores for the second sentence is low because the tail entity type is not a work location in the input text. Also the fact of the confidence score sentence is false.
The confidence scores for the third sentence is low because the tail entity type is not a genre in the input text. Also the fact of the confidence score sentence is false.

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
refined relation labels: {entity_types}"""

# Zero-shot version
CONFIDENCE_SCORE_PROMPT_FEWREL_JOINT_ZERO_SHOT= """Given the 'input text', 'head entity', 'tail entity', 'entity types', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels.
DO NOT ADD ANY OTHER ADDITIONAL WORDS SUCH AS ADVERBS OR ADJECTIVES TO THE RELATIONSHIP CONFIDENCE SCORE SENTENCE.

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
refined relation labels: {entity_types}"""

CONFIDENCE_SCORE_PROMPT_WIKI_SEP = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:
Example input text: Steve Jobs lived in San Francisco, for 10 years.
Example head entity: Steve Jobs
Example tail entity: San Francisco
Example entity types:
Steve Jobs is a Person, and San Francisco is a Location
Example refined relation labels:
1. residence (most relevant)
2. league (second most relevant)
3. cast member (third most relevant)
Example Relationship Confidence Scores Sentences:
1. Person Steve Jobs has been resided in Location San Francisco with a confidence level of 80 - 100%.
2. Person Steve Jobs has been leagued in Location San Francisco with a confidence level of 20 - 30%.
3. Person Steve Jobs has been casted in Location San Francisco with a confidence level of 0 - 20%.

Example Explanation:
The confidence score for the first sentence is high because the head entity type is a person and the tail entity type is location. Also the fact of the confidence score sentence is true and it is mentioned in the input text that Steve Jobs lived in San Francisco.
The confidence scores for the second sentence is low because the tail entity type is not a league in the input text. Also the fact of the confidence score sentence is false.
The confidence scores for the third sentence is low because the tail entity type is not a cast member in the input text. Also the fact of the confidence score sentence is false.

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types: {entity_types}
refined relation labels: {refined_relation_labels}"""

# Zero-shot version
CONFIDENCE_SCORE_PROMPT_WIKI_SEP_ZERO_SHOT = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types: {entity_types}
refined relation labels: {refined_relation_labels}"""

CONFIDENCE_SCORE_PROMPT_WIKI_JOINT = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:
Example input text: Steve Jobs lived in San Francisco, for 10 years.
Example head entity: Steve Jobs
Example tail entity: San Francisco
Example entity types and refined relation labels:
Steve Jobs is a Person, and San Francisco is a Location

1. residence (most relevant)
2. league (second most relevant)
3. cast member (third most relevant)
Example Relationship Confidence Scores Sentences:
1. Person Steve Jobs has been resided in Location San Francisco with a confidence level of 80 - 100%.
2. Person Steve Jobs has been leagued in Location San Francisco with a confidence level of 20 - 30%.
3. Person Steve Jobs has been casted in Location San Francisco with a confidence level of 0 - 20%.

Example Explanation:
The confidence score for the first sentence is high because the head entity type is a person and the tail entity type is location. Also the fact of the confidence score sentence is true and it is mentioned in the input text that Steve Jobs lived in San Francisco.
The confidence scores for the second sentence is low because the tail entity type is not a league in the input text. Also the fact of the confidence score sentence is false.
The confidence scores for the third sentence is low because the tail entity type is not a cast member in the input text. Also the fact of the confidence score sentence is false.

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types and refined relation labels: {entity_types}"""

# Zero-shot version
CONFIDENCE_SCORE_PROMPT_WIKI_JOINT_ZERO_SHOT = """Given the 'input text', 'head entity', 'tail entity', and 'refined relation labels', create sentences that convey the relationship confidence score for each of the three refined relation labels:

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
entity types and refined relation labels: {entity_types}"""

RELATION_EXTRACTION_PROMPT = """Given the 'input text', 'head entity', 'tail entity', 'refined relation labels', and 'relationship confidence scores', determine the most appropriate relationship between the head entity and tail entity.
Example: the most appropriate relationship is: org:city_of_headquarters

input text: {input_text}

head entity: {head_entity}

tail entity: {tail_entity}

refined relation labels: {refined_relation_labels}

relationship confidence scores: {relationship_confidence_scores}"""

RELATION_EXTRACTION_PROMPT_FEWREL= """Given the 'input text', 'head entity', 'tail entity', 'refined relation labels', and 'relationship confidence scores', determine the most appropriate relationship between the head entity and tail entity.
Example: the most appropriate relationship is: performer

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
refined relation labels: {refined_relation_labels}
relationship confidence scores: {relationship_confidence_scores}"""

RELATION_EXTRACTION_PROMPT_WIKI= """Given the 'input text', 'head entity', 'tail entity', 'refined relation labels', and 'relationship confidence scores', determine the most appropriate relationship between the head entity and tail entity.
Example: the most appropriate relationship is: residence

input text: {input_text}
head entity: {head_entity}
tail entity: {tail_entity}
refined relation labels: {refined_relation_labels}
relationship confidence scores: {relationship_confidence_scores}"""
