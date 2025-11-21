from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

# Shared base models and response wrappers
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: str = "admin"  # admin, manager, agent

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Leads
class LeadBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    budget_min: Optional[float] = 0
    budget_max: Optional[float] = 0
    source: Optional[str] = None
    status: str = "new"  # new, qualified, engaged, lost, won
    score: Optional[int] = 0
    notes: Optional[str] = None

class LeadCreate(LeadBase):
    pass

class Lead(LeadBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Investors
class InvestorPreference(BaseModel):
    locations: List[str] = []
    property_types: List[str] = []
    budget_min: float = 0
    budget_max: float = 0
    timeline: Optional[str] = None

class InvestorBase(BaseModel):
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    country: Optional[str] = None
    preferences: InvestorPreference = InvestorPreference()
    notes: Optional[str] = None

class InvestorCreate(InvestorBase):
    pass

class Investor(InvestorBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Properties
class PropertyFinancials(BaseModel):
    acquisition_price: float = 0
    renovation_budget: float = 0
    marketing_cost: float = 0
    expected_arv: float = 0
    roi: float = 0

class PropertyBase(BaseModel):
    title: str
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    stage: str = "acquisition"  # acquisition, renovation, marketing, sold
    description: Optional[str] = None
    financials: PropertyFinancials = PropertyFinancials()

class PropertyCreate(PropertyBase):
    pass

class Property(PropertyBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Renovation Tasks
class Contractor(BaseModel):
    name: str
    company: Optional[str] = None
    phone: Optional[str] = None

class RenovationTaskBase(BaseModel):
    property_id: str
    title: str
    status: str = "todo"  # todo, in_progress, review, done
    assignee: Optional[Contractor] = None
    budget: float = 0
    spend: float = 0
    progress: int = 0
    due_date: Optional[datetime] = None

class RenovationTaskCreate(RenovationTaskBase):
    pass

class RenovationTask(RenovationTaskBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Communications
class CommunicationBase(BaseModel):
    entity_type: str  # investor, lead, property, task
    entity_id: str
    channel: str  # email, phone, whatsapp, note, system
    message: str
    author: Optional[str] = None

class CommunicationCreate(CommunicationBase):
    pass

class Communication(CommunicationBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# Activity Logs
class ActivityLogBase(BaseModel):
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    meta: dict = {}

class ActivityLogCreate(ActivityLogBase):
    pass

class ActivityLog(ActivityLogBase):
    id: str = Field(..., alias="_id")
    created_at: Optional[datetime]

    class Config:
        allow_population_by_field_name = True

# AI
class AIScoreLeadInput(BaseModel):
    profile: dict

class AIPropertyDescInput(BaseModel):
    attributes: dict

class AISummarizeRenovationInput(BaseModel):
    tasks: List[dict]

class AIGenerateInvestorUpdateInput(BaseModel):
    property: dict
    investor: dict

class AIRecommendNextActionInput(BaseModel):
    entityType: str
    entityData: dict

class AISummarizeCommunicationsInput(BaseModel):
    logs: List[dict]

class AICalcInvestmentScoreInput(BaseModel):
    property: dict
