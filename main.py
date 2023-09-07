from fastapi import FastAPI
import uvicorn, joblib  # noqa: E401
from pydantic import BaseModel

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = FastAPI()

label_encoding_dict = {'Good': 0, 'Poor': 1, 'Standard': 2}
encoded_labels = {v: k for k,v in label_encoding_dict.items()}

class request_body(BaseModel):
    annual_income: float
    num_credit_card: int
    interest_rate: int
    delay_from_due_date: int
    credit_mix: float
    monthly_balance: float
    credit_history_age: int
    num_credit_inquiries: float
    changed_credit_limit: float
    num_of_delayed_payment: int
    outstanding_debt: float
    monthly_inhand_salary: float
    payment_of_min_amount: int
    uncommon_loan_type: int
    total_emi_per_month: float
    payment_behaviour: int
    
@app.get("/")
def home():
    return {"message": "Hello, welcome to the Credit Score Prediction API!"}

@app.post("/predict")
def predict(data: request_body):
    test_data = [[
        data.annual_income,
        data.num_credit_card,
        data.interest_rate,
        data.delay_from_due_date,
        data.credit_mix,
        data.monthly_balance,
        data.credit_history_age,
        data.num_credit_inquiries,
        data.changed_credit_limit,
        data.num_of_delayed_payment,
        data.outstanding_debt,
        data.monthly_inhand_salary,
        data.payment_of_min_amount,
        data.uncommon_loan_type,
        data.total_emi_per_month,
        data.payment_behaviour
    ]]
    
    scaled_test_data = scaler.transform(test_data)
    pred = encoded_labels[model.predict(scaled_test_data)[0]]
    pred = pred.lower()
    return {"result": f"Your credit score is {pred}."}
    
if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1',port=8000)