from pydantic import BaseModel

# Description of inputs for validation
class MCQModel(BaseModel):
    full_text: str
