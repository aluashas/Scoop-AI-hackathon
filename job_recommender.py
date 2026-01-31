from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    skills: List[str]
    desired_titles: List[str]
    location: Optional[str] = None
    remote_only: bool = False
    min_compensation: Optional[float] = None
    experience_level: Optional[str] = None
    industries: Optional[List[str]] = None


@dataclass(frozen=True)
class JobPosting:
    id: str
    title: str
    company: str
    description: str
    skills: List[str]
    location: Optional[str] = None
    remote: bool = False
    compensation: Optional[float] = None
    experience_level: Optional[str] = None
    industry: Optional[str] = None


@dataclass(frozen=True)
class Recommendation:
    job: JobPosting
    score: float
    rationale: str


def _normalize(tokens: Iterable[str]) -> List[str]:
    return sorted(
        {token.strip().lower() for token in tokens if token and token.strip()}
    )


def _score_job(profile: CandidateProfile, job: JobPosting) -> Recommendation:
    profile_skills = set(_normalize(profile.skills))
    job_skills = set(_normalize(job.skills))
    overlap = profile_skills.intersection(job_skills)

    score = 0.0
    rationale_bits: List[str] = []

    if overlap:
        overlap_ratio = len(overlap) / max(len(job_skills), 1)
        score += overlap_ratio * 60
        rationale_bits.append(f"{len(overlap)} skill matches")

    desired_titles = set(_normalize(profile.desired_titles))
    if desired_titles:
        title_tokens = set(_normalize(job.title.split()))
        title_overlap = desired_titles.intersection(title_tokens)
        if title_overlap:
            score += 15
            rationale_bits.append("title matches target")

    if profile.remote_only:
        if job.remote:
            score += 10
            rationale_bits.append("remote friendly")
        else:
            score -= 20
            rationale_bits.append("not remote")

    if profile.location and job.location:
        if profile.location.lower() in job.location.lower():
            score += 10
            rationale_bits.append("location match")

    if profile.min_compensation and job.compensation:
        if job.compensation >= profile.min_compensation:
            score += 10
            rationale_bits.append("meets compensation floor")
        else:
            score -= 10
            rationale_bits.append("below compensation target")

    if profile.experience_level and job.experience_level:
        if profile.experience_level.lower() == job.experience_level.lower():
            score += 5
            rationale_bits.append("experience level match")

    if profile.industries and job.industry:
        if job.industry.lower() in _normalize(profile.industries):
            score += 5
            rationale_bits.append("industry preference")

    if not rationale_bits:
        rationale_bits.append("general fit")

    return Recommendation(
        job=job, score=round(score, 2), rationale="; ".join(rationale_bits)
    )


def recommend_jobs(
    profile: CandidateProfile,
    jobs: Iterable[JobPosting],
    top_k: int = 5,
    use_gpt: Optional[bool] = None,
) -> List[Recommendation]:
    if use_gpt is None:
        use_gpt = os.getenv("USE_GPT_RECOMMENDER", "false").lower() == "true"

    jobs_list = list(jobs)
    if use_gpt and os.getenv("OPENAI_API_KEY"):
        try:
            gpt_recs = _gpt_rank(profile, jobs_list, top_k)
            if gpt_recs:
                return gpt_recs
        except Exception:
            pass

    scored = [_score_job(profile, job) for job in jobs_list]
    scored.sort(key=lambda rec: rec.score, reverse=True)
    return scored[:top_k]


def _gpt_rank(
    profile: CandidateProfile,
    jobs: List[JobPosting],
    top_k: int,
) -> List[Recommendation]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return []

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "profile": profile.__dict__,
        "jobs": [job.__dict__ for job in jobs],
        "top_k": top_k,
    }

    prompt = (
        "You are a job recommendation engine. Rank jobs for the candidate. "
        "Return JSON with keys: recommendations (list of {id, score, rationale}). "
        "Only return valid JSON."
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        text = response.output_text
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
        )
        text = response.choices[0].message.content or ""

    try:
        data = json.loads(text)
    except Exception:
        return []

    recommendations = []
    lookup = {job.id: job for job in jobs}
    for item in data.get("recommendations", []):
        job = lookup.get(item.get("id"))
        if not job:
            continue
        recommendations.append(
            Recommendation(
                job=job,
                score=float(item.get("score", 0)),
                rationale=str(item.get("rationale", "")),
            )
        )

    return recommendations[:top_k]
