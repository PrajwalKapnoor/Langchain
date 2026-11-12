from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Give information about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser

response = chain.invoke({"topic": "Python"})

print(response)
print(type(response))
