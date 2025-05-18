from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import BerthAllocationModel

app = FastAPI()

# Define request body
class InputData(BaseModel):
    features: list[float]

# Load model
model = BerthAllocationModel()
model.load_state_dict(torch.load("berth_allocation_model.pth"))
model.eval()

@app.post("/")
def predict(data: InputData):
    try:
        input_tensor = torch.tensor([data.features], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor).squeeze().tolist()
        return {
            "eta": output[0],
            "etd": output[1],
            "berth": int(output[2])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
