"""
FastAPI application entrypoint for Clinical Platform API
"""
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Clinical Platform API", version="1.0.0")
app.include_router(router)
