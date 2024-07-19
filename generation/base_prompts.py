BASE_PROMPT = """
You are reading text extracted from a PDF with several pages. The pages are divided by a line saying 'NEW PAGE'. 
Your role is to {role_description}. If the type of questions requested are impossible to generate due to the simplicity of the document, default to simpler factual questions.
The PDFs might contain tables or images that are poorly parsed in the text. Avoid asking questions about these.
If the text seems to only contain uninteresting information, output "unanswerable" as the answer.
Here are some examples for questions that follow your role:
{examples}
"""

BASE_USER_CONTENT = """The text contained in the PDF is: 
{text} 

Create the question answer pairs following this format:
Q#: 
A#:

If you can't generate a questions for the text, write "unanswerable" as the answer.
"""

PROMPTS = [
    {
        "role_description": "understand the content of the PDF and create as many pairs of questions and answers as you need to cover the content of the PDF comprehensively. The questions should be varied, covering factual information, inferences, and deeper analysis of the text.",
        "examples": """
        Q1: What is the main topic of the document?
        A1: The main topic of the document is...
        
        Q2: What are the key points discussed in the first section?
        A2: The key points discussed in the first section include...

        Q3: How does the author support their argument about X?
        A3: The author supports their argument about X by...

        Q4: What can be inferred about Y from the document?
        A4: From the document, it can be inferred that Y...

        Q5: What are the implications of Z mentioned in the document?
        A5: The implications of Z mentioned in the document are...
        """
    },
    {
        "role_description": "focus on generating enough pairs of questions and answers for each section of the document to ensure a detailed and complete coverage the document.",
        "examples": """
        Q1: What is the primary focus of the first section?
        A1: The primary focus of the first section is...

        Q2: What are the significant details mentioned in the second section?
        A2: The significant details mentioned in the second section include...

        Q3: How does the information in the third section relate to the overall topic of the document?
        A3: The information in the third section relates to the overall topic by...
        """
    },
    {
        "role_description": "understand the content of the PDF and create as many pairs of questions and answers as you need to cover the content of the PDF comprehensively. The questions should require critical thinking and analysis.",
        "examples": """
        Q1: What arguments does the author present in support of their thesis?
        A1: The arguments presented by the author in support of their thesis include...

        Q2: How does the author compare X and Y in the text?
        A2: The author compares X and Y by...

        Q3: What are the potential implications of the findings discussed in the document?
        A3: The potential implications of the findings are...
        """
    },
    {
        "role_description": "create as many pairs of questions and answers as you need to cover both summaries of sections and specific details. Ensure a coverage of broad themes and granular information.",
        "examples": """
        Q1: What is the summary of the first section?
        A1: The summary of the first section is...

        Q2: What specific data or evidence is provided in the second section?
        A2: The specific data or evidence provided in the second section includes...

        Q3: How do the details in the third section support the main argument of the document?
        A3: The details in the third section support the main argument by...
        """
    },
    {
        "role_description": "understand the content of the PDF and create as many pairs of questions and answers as you need to cover the content of the PDF comprehensively. The questions should be varied, covering factual information, inferences, and deeper analysis of the text. The questions should be asked in a general manner without introducing details from the document itself.",
        "examples": """
        Q1: What is the summary of the first section?
        A1: The first section, called xxx, can be summarized as is...

        Q2: What specific data or evidence is provided in the second section?
        A2: In the section called xxx, there is a much data and evidence presented, such as...

        Q3: How do the details in the third section support the main argument of the document?
        A3: The details in the section on "xxx" support the main argument by...
        """
    }
]

def create_prompts(text):
    prompts = []
    for prompt in PROMPTS:
        system_content = BASE_PROMPT.format(
            role_description=prompt["role_description"],
            examples=prompt["examples"]
        )
        prompts.append([
            {"role": "system", "content": system_content},
            {"role": "user", "content": BASE_USER_CONTENT.format(text=text)}
        ])
    return prompts