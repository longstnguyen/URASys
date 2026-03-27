FAQ_DETAIL_EXPANSION_PROMPT_TEMPLATE = """
# TASK: Vietnamese FAQ Detail Expansion

## ROLE
You are an expert in refining and expanding FAQ content. Your task is to analyze a given FAQ pair and extract detailed information to create additional, nuanced FAQ pairs.

---
## GOAL
Your task is to generate up to {max_new_faq_pairs} new FAQ pairs derived from the provided input FAQ pair. Each new FAQ pair should offer more detailed and specific insights related to the original FAQ content. The final output must be in Vietnamese.

---
## GUIDELINES
1. **Analysis of the Input FAQ:**
   * Carefully analyze the provided FAQ pair to identify any additional details, clarifications, or aspects that can be turned into new FAQ pairs.
   * Focus on expanding the details by considering underlying reasons, examples, or related subtopics.
2. **Language Requirement:**
   * The input FAQ pair is in Vietnamese, and your output must be entirely in Vietnamese.
3. **Internal Reasoning Process:**
   * Use a step-by-step internal reasoning process (chain-of-thought) to determine the most relevant additional FAQ pairs. 
   * **Important:** Do not include any internal chain-of-thought or reasoning steps in your final output.
4. **Output Format:**
   * Return the final output in strict JSON format as an array of objects. Each object must include two keys: "question" and "answer". For example:
    [
        {{
            "question": "Question in Vietnamese?",
            "answer": "Answer in Vietnamese."
        }},
        ...
    ]
5. **Relevance and Specificity:**
   * Ensure that each generated FAQ pair adds new, specific details that naturally extend the input FAQ pair.
   * Avoid redundancy and make sure the new FAQ pairs are directly related to the original content.

---
## EXAMPLE
Input FAQ Pair:
{{
    "question": "Who was the first person to walk on the Moon?",
    "answer": "The first person to walk on the Moon was Neil Armstrong."
}}

Expected Output:
[
    {{
        "question": "In what year did Neil Armstrong walk on the Moon?",
        "answer": "Neil Armstrong walked on the Moon in 1969."
    }},
    {{
        "question": "What were the key milestones in Neil Armstrong's career?",
        "answer": "Neil Armstrong was a celebrated astronaut who participated in multiple space missions and made significant contributions to aerospace exploration."
    }}
]

---
## ADDITIONAL INSTRUCTIONS
* Remember to keep your output within the limit of **{max_new_faq_pairs} new FAQ pairs**.
* Do not include any internal chain-of-thought or reasoning steps in your final output.

---------------------
##### REAL DATA #####
---------------------
REAL DATA:
FAQ Pair: {faq_pair}
Output:
"""
