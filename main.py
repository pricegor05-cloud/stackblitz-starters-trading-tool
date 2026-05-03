from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# allow StackBlitz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/scan")
def scan():
    return {
        "TSLA": {
            "price": round(200 + random.random()*50, 2),
            "signal": "CALL"
        },
        "AAPL": {
            "price": round(150 + random.random()*10, 2),
            "signal": "HOLD"
        }
    }