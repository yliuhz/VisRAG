
SYSTEM_PROMPT='''

You're a very effective entity extraction system. Please extract all named entities that are important for solving the questions below. Return output as a well-formed JSON-formatted string with the following format:
{{
    "named_entities": [
        {{
            "name": "First for Women"
        }}, 
        {{
            "name": "Arthur's Magazine"
        }}
    ]
}}

# Example
Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: 
{{
    "named_entities": [
        {{
            "name": "First for Women"
        }}, 
        {{
            "name": "Arthur's Magazine"
        }}
    ]
}}
'''

USER_PROMPT='''

# Real Data
Question: {}
Answer: 
'''
