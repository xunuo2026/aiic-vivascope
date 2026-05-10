from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TrainingMode = Literal["knowledge", "project", "mixed"]
InterviewLength = Literal["short", "standard", "deep"]
InterviewScenario = Literal["保研复试", "科研项目答辩", "课程项目展示", "实验室面试", "技术类实习面试"]
FollowupStyle = Literal["温和学长型", "严格导师型", "压力追问型"]
SessionStatus = Literal["interview", "finished"]


class StartRequest(BaseModel):
    scenario: InterviewScenario
    major: str = Field(min_length=2, max_length=120)
    project: str = Field(default="", max_length=6000)
    focus: str = Field(default="", max_length=800)
    style: FollowupStyle
    mode: TrainingMode
    interview_length: InterviewLength = "standard"


class ProjectMap(BaseModel):
    theme: str
    motivation: str
    methods: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    contribution: list[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    dimension: str
    level: int = Field(ge=1, le=5)
    reason: str


class KnowledgePoint(BaseModel):
    name: str
    why_relevant: str
    probe_example: str


class Feedback(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    answer_frame: list[str] = Field(default_factory=list)
    score: int = Field(ge=0, le=100)


class Turn(BaseModel):
    round_index: int
    question: str
    answer: str
    feedback: Feedback


class FinalReport(BaseModel):
    total_score: int = Field(ge=0, le=100)
    answer_summary: str = ""
    most_vulnerable_project_points: list[str] = Field(default_factory=list)
    knowledge_weaknesses: list[str] = Field(default_factory=list)
    expression_issues: list[str] = Field(default_factory=list)
    next_training_tasks: list[str] = Field(default_factory=list)
    overall_advice: str = ""
    closing_comment: str


class SessionState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus = "interview"
    max_rounds: int = 6
    input: StartRequest
    project_map: ProjectMap | None = None
    risk_radar: list[RiskItem] = Field(default_factory=list)
    knowledge_points: list[KnowledgePoint] = Field(default_factory=list)
    current_question: str
    turns: list[Turn] = Field(default_factory=list)
    final_report: FinalReport | None = None
    ai_source: str = "dashscope"
    ai_warning: str = ""


class StartResponse(BaseModel):
    session: SessionState


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=5000)


class AnswerResponse(BaseModel):
    session: SessionState
    turn: Turn
    next_question: str | None = None
    final_report: FinalReport | None = None


class RestoreRequest(BaseModel):
    session: SessionState
