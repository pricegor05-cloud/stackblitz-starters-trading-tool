from alpha_engine import alpha_engine
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

    market = get_market_data()

    signals = alpha_engine(market)

    return signals