from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import os
import httpx

from schemas import (
    Token, TokenData,
    UserCreate, UserLogin, User,
    LeadCreate, Lead,
    InvestorCreate, Investor,
    PropertyCreate, Property,
    RenovationTaskCreate, RenovationTask,
    CommunicationCreate, Communication,
    ActivityLogCreate, ActivityLog,
    AIScoreLeadInput, AIPropertyDescInput, AISummarizeRenovationInput,
    AIGenerateInvestorUpdateInput, AIRecommendNextActionInput, AISummarizeCommunicationsInput,
    AICalcInvestmentScoreInput
)

# Database helpers (provided by pre-configured environment)
from database import db, create_document, get_documents

app = FastAPI(title="Real Estate Automation SaaS API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth setup
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Utils

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


async def get_user_by_email(email: str) -> Optional[dict]:
    users = await get_documents("user", {"email": email}, limit=1)
    return users[0] if users else None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    users = await get_documents("user", {"_id": user_id}, limit=1)
    if not users:
        raise credentials_exception
    return users[0]


# Routes
@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# Auth
@app.post("/api/v1/auth/register", response_model=User)
async def register(user: UserCreate):
    existing = await get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = user.dict()
    doc.update({
        "password": get_password_hash(user.password),
        "created_at": datetime.utcnow(),
    })
    created = await create_document("user", doc)
    created["password"] = ""
    return created


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user.get("password", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": user["_id"], "role": user.get("role", "admin")})
    return {"access_token": access_token, "token_type": "bearer"}


# Generic helpers
async def list_collection(name: str, limit: int = 100):
    return await get_documents(name, {}, limit)


# Leads
@app.post("/api/v1/leads", response_model=Lead)
async def create_lead(payload: LeadCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("lead", doc)


@app.get("/api/v1/leads", response_model=List[Lead])
async def get_leads(user=Depends(get_current_user)):
    return await list_collection("lead")


# Investors
@app.post("/api/v1/investors", response_model=Investor)
async def create_investor(payload: InvestorCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("investor", doc)


@app.get("/api/v1/investors", response_model=List[Investor])
async def get_investors(user=Depends(get_current_user)):
    return await list_collection("investor")


# Properties
@app.post("/api/v1/properties", response_model=Property)
async def create_property(payload: PropertyCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("property", doc)


@app.get("/api/v1/properties", response_model=List[Property])
async def get_properties(user=Depends(get_current_user)):
    return await list_collection("property")


# Renovation Tasks
@app.post("/api/v1/tasks", response_model=RenovationTask)
async def create_task(payload: RenovationTaskCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("renovationtask", doc)


@app.get("/api/v1/tasks", response_model=List[RenovationTask])
async def get_tasks(user=Depends(get_current_user)):
    return await list_collection("renovationtask")


# Communications
@app.post("/api/v1/comms", response_model=Communication)
async def create_comm(payload: CommunicationCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("communication", doc)


@app.get("/api/v1/comms", response_model=List[Communication])
async def get_comms(user=Depends(get_current_user)):
    return await list_collection("communication")


# Activity Logs
@app.post("/api/v1/activity", response_model=ActivityLog)
async def log_activity(payload: ActivityLogCreate, user=Depends(get_current_user)):
    doc = payload.dict()
    doc.update({"created_at": datetime.utcnow(), "owner_id": user["_id"]})
    return await create_document("activitylog", doc)


@app.get("/api/v1/activity", response_model=List[ActivityLog])
async def get_activity(user=Depends(get_current_user)):
    return await list_collection("activitylog")


# AI Engine stubs with model-agnostic routing
AI_BASES = {
    "openai": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    "anthropic": os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com"),
    "groq": os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
}


def choose_ai_provider() -> str:
    # naive preference order via available API keys
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("ANTHROPIC_API_KEY"): return "anthropic"
    if os.getenv("GROQ_API_KEY"): return "groq"
    return "mock"


async def ai_request(payload: dict, stream: bool = False):
    provider = choose_ai_provider()
    if provider == "mock":
        return {"provider": provider, "result": "mocked", "payload": payload}
    headers = {}
    url = ""
    if provider == "openai":
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        url = f"{AI_BASES['openai']}/chat/completions"
        body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages": payload.get("messages", []), "stream": stream}
    elif provider == "anthropic":
        headers = {"x-api-key": os.getenv('ANTHROPIC_API_KEY'), "anthropic-version": "2023-06-01"}
        url = f"{AI_BASES['anthropic']}/v1/messages"
        body = {"model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"), "messages": payload.get("messages", [])}
    else:
        headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
        url = f"{AI_BASES['groq']}/chat/completions"
        body = {"model": os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"), "messages": payload.get("messages", []), "stream": stream}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


@app.post("/api/v1/ai/scoreLead")
async def ai_score_lead(payload: AIScoreLeadInput, user=Depends(get_current_user)):
    profile = payload.profile
    # simple heuristic + AI fallback
    base = 0
    budget = profile.get("budget", 0)
    country = (profile.get("country") or "").lower()
    if budget >= 100000: base += 30
    if country in ["singapore", "uae", "china", "vietnam"]: base += 20
    if profile.get("timeline") in ["immediate", "3-6 months"]: base += 20
    base = min(90, base)
    ai = await ai_request({"messages": [{"role": "user", "content": f"Score this lead 0-100 based on: {profile}"}]})
    return {"score": min(100, base + 10), "ai": ai}


@app.post("/api/v1/ai/generatePropertyDescription")
async def ai_property_desc(payload: AIPropertyDescInput, user=Depends(get_current_user)):
    attrs = payload.attributes
    prompt = f"Write a compelling real estate listing in 120 words: {attrs}"
    ai = await ai_request({"messages": [{"role": "user", "content": prompt}]})
    return {"description": ai.get("result") or ai}


@app.post("/api/v1/ai/summarizeRenovation")
async def ai_summarize_reno(payload: AISummarizeRenovationInput, user=Depends(get_current_user)):
    prompt = f"Summarize renovation progress from tasks: {payload.tasks}"
    ai = await ai_request({"messages": [{"role": "user", "content": prompt}]})
    return {"summary": ai.get("result") or ai}


@app.post("/api/v1/ai/generateInvestorUpdate")
async def ai_investor_update(payload: AIGenerateInvestorUpdateInput, user=Depends(get_current_user)):
    prompt = f"Draft a brief investor update for {payload.investor} about property {payload.property}"
    ai = await ai_request({"messages": [{"role": "user", "content": prompt}]})
    return {"update": ai.get("result") or ai}


@app.post("/api/v1/ai/recommendNextAction")
async def ai_next_action(payload: AIRecommendNextActionInput, user=Depends(get_current_user)):
    prompt = f"Recommend next action for {payload.entityType}: {payload.entityData}"
    ai = await ai_request({"messages": [{"role": "user", "content": prompt}]})
    return {"action": ai.get("result") or ai}


@app.post("/api/v1/ai/summarizeCommunications")
async def ai_sum_comms(payload: AISummarizeCommunicationsInput, user=Depends(get_current_user)):
    prompt = f"Summarize communications: {payload.logs}"
    ai = await ai_request({"messages": [{"role": "user", "content": prompt}]})
    return {"summary": ai.get("result") or ai}


@app.post("/api/v1/ai/calculateInvestmentScore")
async def ai_invest_score(payload: AICalcInvestmentScoreInput, user=Depends(get_current_user)):
    p = payload.property
    score = 0
    arv = p.get("financials", {}).get("expected_arv", 0)
    acquisition = p.get("financials", {}).get("acquisition_price", 1)
    roi = 0
    if acquisition:
        roi = (arv - acquisition) / acquisition * 100
    if roi > 20: score += 40
    if p.get("stage") == "marketing": score += 10
    ai = await ai_request({"messages": [{"role": "user", "content": f"Score investment attractiveness: {p}"}]})
    return {"score": min(100, score + 10), "roi": roi, "ai": ai}


# Automation Webhooks (incoming)
@app.post("/api/v1/webhooks/lead-intake")
async def webhook_lead_intake(payload: dict, x_signature: Optional[str] = Header(None)):
    # For demo, skip signature validation; production should HMAC validate
    data = payload.copy()
    data.update({"created_at": datetime.utcnow(), "source": data.get("source", "webhook")})
    created = await create_document("lead", data)
    await create_document("activitylog", {"action": "lead_intake", "entity_type": "lead", "entity_id": created["_id"], "created_at": datetime.utcnow()})
    return {"status": "received", "lead_id": created["_id"]}


@app.post("/api/v1/webhooks/contractor-update")
async def webhook_contractor_update(payload: dict):
    await create_document("activitylog", {"action": "contractor_update", "entity_type": "task", "meta": payload, "created_at": datetime.utcnow()})
    return {"ok": True}


@app.post("/api/v1/webhooks/property-status")
async def webhook_property_status(payload: dict):
    await create_document("activitylog", {"action": "property_status", "entity_type": "property", "meta": payload, "created_at": datetime.utcnow()})
    return {"ok": True}


@app.post("/api/v1/webhooks/investor-communication")
async def webhook_investor_comm(payload: dict):
    await create_document("communication", {"entity_type": "investor", "entity_id": payload.get("investor_id"), "channel": payload.get("channel", "system"), "message": payload.get("message", ""), "created_at": datetime.utcnow()})
    return {"ok": True}


# Outgoing notifications (stubs)
@app.post("/api/v1/automations/weekly-investor-report")
async def weekly_investor_report():
    await create_document("activitylog", {"action": "weekly_investor_report", "entity_type": "investor", "created_at": datetime.utcnow()})
    return {"ok": True}


@app.post("/api/v1/automations/lead-followup-reminder")
async def lead_followup_reminder():
    await create_document("activitylog", {"action": "lead_followup_reminder", "entity_type": "lead", "created_at": datetime.utcnow()})
    return {"ok": True}


@app.post("/api/v1/automations/contractor-task-alert")
async def contractor_task_alert():
    await create_document("activitylog", {"action": "contractor_task_alert", "entity_type": "task", "created_at": datetime.utcnow()})
    return {"ok": True}


@app.post("/api/v1/automations/pipeline-status-notify")
async def pipeline_status_notify():
    await create_document("activitylog", {"action": "pipeline_status_notify", "entity_type": "property", "created_at": datetime.utcnow()})
    return {"ok": True}


# Seed endpoint for demo data
@app.post("/api/v1/seed")
async def seed():
    now = datetime.utcnow()
    admin = await create_document("user", {"email": "admin@estateops.io", "name": "Admin", "role": "admin", "password": get_password_hash("admin123"), "created_at": now})
    # basic samples; full dataset will be in documentation
    await create_document("lead", {"name": "Li Wei", "email": "li.wei@example.cn", "country": "China", "budget_min": 200000, "budget_max": 800000, "score": 75, "created_at": now})
    await create_document("investor", {"name": "Nguyen Anh", "email": "anh@vn.example", "country": "Vietnam", "preferences": {"locations": ["Hanoi"], "budget_min": 150000, "budget_max": 500000}, "created_at": now})
    await create_document("property", {"title": "Marina Bay View Condo", "city": "Singapore", "country": "Singapore", "stage": "marketing", "financials": {"acquisition_price": 900000, "renovation_budget": 60000, "expected_arv": 1100000}, "created_at": now})
    await create_document("renovationtask", {"property_id": "sample", "title": "Kitchen remodel", "status": "in_progress", "budget": 20000, "spend": 8000, "progress": 40, "created_at": now})
    return {"ok": True}


@app.get("/test")
async def test_db():
    # Provided by preconfigured database module
    colls = ["user", "lead", "investor", "property", "renovationtask", "communication", "activitylog"]
    return {"collections": colls}
