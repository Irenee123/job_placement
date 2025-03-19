from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, confloat
import joblib
import numpy as np

# Load the best model saved in Task 1
best_model = joblib.load('best_model.pkl')

# Define the input data model using Pydantic
class PlacementRequest(BaseModel):
    cgpa: confloat(ge=0.0, le=10.0)  # CGPA must be between 0.0 and 10.0
    resume_score: confloat(ge=0.0, le=10.0)  # Resume score must be between 0.0 and 10.0

# Initialize FastAPI app
app = FastAPI(title="Student Placement Prediction API", version="1.0")

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PlacementRequest):
    try:
        # Prepare input data
        input_data = np.array([[request.cgpa, request.resume_score]])
        
        # Make prediction
        prediction = best_model.predict(input_data)
        
        # Return the result
        placement = "Placed" if prediction[0] >= 0.5 else "Not Placed"
        return {"prediction": placement}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)