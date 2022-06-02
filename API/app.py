from fastapi import FastAPI
from MCQ import MCQ
from MCQModel import MCQModel

app = FastAPI()
mcq = MCQ()


@app.get("/{full_text}")
def get_MCQ(full_text: str):
    return mcq.run_pipeline(full_text)
