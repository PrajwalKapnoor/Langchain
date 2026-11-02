from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Student(BaseModel):
    name: str = Field(description="Student name")
    age: int = Field(description="Student age")
    skills: list[str] = Field(description="Student skills")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_model = model.with_structured_output(Student)

response = structured_model.invoke(
    "My name is Rahul. I am 21 years old. I know Python, SQL and Pandas."
)

print(response)
print(response.name)
print(response.age)
print(response.skills)
