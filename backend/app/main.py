from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.database import SessionLocal, init_db
from app.db.seed_data import seed_demo_employees


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    init_db()
    with SessionLocal() as db:
        seed_demo_employees(db)
    yield


app = FastAPI(title="Secure Virtual Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
