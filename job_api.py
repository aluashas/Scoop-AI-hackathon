from __future__ import annotations
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from job_recommender import CandidateProfile, JobPosting, Recommendation, recommend_jobs


class CandidateProfileIn(BaseModel):
    name: str
    skills: List[str] = Field(default_factory=list)
    desired_titles: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    remote_only: bool = False
    min_compensation: Optional[float] = None
    experience_level: Optional[str] = None
    industries: Optional[List[str]] = None


class JobPostingIn(BaseModel):
    id: str
    title: str
    company: str
    description: str
    skills: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    remote: bool = False
    compensation: Optional[float] = None
    experience_level: Optional[str] = None
    industry: Optional[str] = None


class RecommendationOut(BaseModel):
    id: str
    title: str
    company: str
    score: float
    rationale: str


class RecommendationRequest(BaseModel):
    profile: CandidateProfileIn
    jobs: List[JobPostingIn]
    top_k: int = 5
    use_gpt: Optional[bool] = None


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationOut]


app = FastAPI(title="SALA Job Recommendations", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(payload: RecommendationRequest) -> RecommendationResponse:
    profile = CandidateProfile(**payload.profile.model_dump())
    jobs = [JobPosting(**job.model_dump()) for job in payload.jobs]
    recs: List[Recommendation] = recommend_jobs(
        profile,
        jobs,
        top_k=payload.top_k,
        use_gpt=payload.use_gpt,
    )

    return RecommendationResponse(
        recommendations=[
            RecommendationOut(
                id=rec.job.id,
                title=rec.job.title,
                company=rec.job.company,
                score=rec.score,
                rationale=rec.rationale,
            )
            for rec in recs
        ]
    )
