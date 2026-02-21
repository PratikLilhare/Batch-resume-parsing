from typing import List, Optional
from pydantic import BaseModel, Field

class Education(BaseModel):
    degree: Optional[str] = Field(None, description="Degree obtained (e.g., B.S. Computer Science)")
    institution: Optional[str] = Field(None, description="University or institution name")
    year: Optional[str] = Field(None, description="Year of graduation or dates attended")

class Experience(BaseModel):
    role: Optional[str] = Field(None, description="Job title or role")
    company: Optional[str] = Field(None, description="Company name")
    duration: Optional[str] = Field(None, description="Dates of employment")
    description: Optional[str] = Field(None, description="Brief summary of responsibilities")

class ResumeData(BaseModel):
    full_name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(default_factory=list, description="List of technical skills, tools, and languages")
    experience: List[Experience] = Field(default_factory=list, description="Work experience history")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    summary: Optional[str] = Field(None, description="Brief professional summary or objective")
