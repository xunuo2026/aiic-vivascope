from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

from app.schemas import ResumePrefill


MAX_RESUME_TEXT_CHARS = 12000
MAX_PREVIEW_CHARS = 800


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    chunks: list[str] = []
    for page in reader.pages[:8]:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
    text = "\n".join(chunks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:MAX_RESUME_TEXT_CHARS]


def heuristic_resume_prefill(text: str) -> ResumePrefill:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}|(?<=。)\s*", text) if part.strip()]
    compact = re.sub(r"\s+", " ", text).strip()
    major = _extract_major(compact)
    project = _extract_project(paragraphs, compact)
    target_profile = _extract_target_profile(compact)
    focus = (
        "根据简历内容，重点训练项目动机、方法选择、实验/实现可靠性、个人贡献边界、"
        "结果解释和与申请方向相关的基础知识。"
    )
    return ResumePrefill(
        major=major[:120],
        target_profile=target_profile[:1200],
        project=project[:6000],
        focus=focus[:800],
        source_text_preview=compact[:MAX_PREVIEW_CHARS],
    )


def resume_prefill_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是问脉 VivaScope 的简历信息提取器。"
                "请从理工科本科生简历文本中提取适合口试训练表单的内容。"
                "只返回严格 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请把下面简历文本结构化为问脉训练表单建议。\n"
                "major 要概括专业、年级、主要技术背景，不超过 120 字。\n"
                "target_profile 要概括申请导师方向、实验室方向、岗位要求或适合面试的目标画像；如果简历没有明确目标，就根据技术经历给出合理训练画像。\n"
                "project 要整理最适合被连续追问的 1-2 段项目经历，包含背景、目标、方法、实验/实现、结果、个人贡献和不足。\n"
                "focus 要写用户最该训练的追问方向。\n"
                "返回格式："
                '{"prefill":{"major":"","target_profile":"","project":"","focus":""}}\n\n'
                f"简历文本：\n{text[:MAX_RESUME_TEXT_CHARS]}"
            ),
        },
    ]


def merge_prefill(fallback: ResumePrefill, result: dict | None) -> ResumePrefill:
    if not result:
        return fallback
    raw = result.get("prefill", result)
    if not isinstance(raw, dict):
        return fallback
    data = fallback.model_dump()
    for key in ("major", "target_profile", "project", "focus"):
        value = str(raw.get(key, "")).strip()
        if value:
            data[key] = value
    return ResumePrefill(**data)


def _extract_major(text: str) -> str:
    match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+专业|专业[:：]\s*[\u4e00-\u9fa5A-Za-z0-9]+)", text)
    if match:
        major = match.group(0).replace("专业:", "专业：")
    else:
        major = "请根据简历补充专业背景、年级和主要技术栈"
    skill_match = re.search(r"(机器学习|深度学习|计算机视觉|人工智能|自动化|电子信息|材料|物理|控制|嵌入式|机器人)", text)
    if skill_match and skill_match.group(0) not in major:
        major = f"{major}，涉及{skill_match.group(0)}相关经历"
    return major


def _extract_target_profile(text: str) -> str:
    target_keywords = "保研|复试|导师|实验室|岗位|实习|求职|研究方向|方向|申请"
    sentences = _pick_sentences(text, target_keywords, limit=4)
    if sentences:
        return "；".join(sentences)
    return "根据简历技术经历，建议围绕目标导师/实验室方向或技术岗位要求补充面试目标，以便生成更有针对性的追问。"


def _extract_project(paragraphs: list[str], text: str) -> str:
    project_keywords = "项目|课题|研究|实验|系统|模型|算法|比赛|论文|实习|YOLO|Transformer|CNN|机器人|检测|分类"
    chosen = [p for p in paragraphs if re.search(project_keywords, p, flags=re.I)]
    if not chosen:
        chosen = _pick_sentences(text, project_keywords, limit=6)
    project = "\n".join(chosen[:4]).strip()
    return project or "请补充一段最希望训练的项目经历：背景、目标、方法、实验/实现、结果、个人贡献和不足。"


def _pick_sentences(text: str, keywords: str, limit: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"[。；;]\s*", text) if part.strip()]
    return [sentence[:220] for sentence in sentences if re.search(keywords, sentence, flags=re.I)][:limit]
