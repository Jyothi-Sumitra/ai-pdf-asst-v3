import hashlib
import json
import os
import tempfile

from .models import ATSAnalysis, ATSResult
from .prompts import ats_prompt


DEFAULT_CACHE_PATH = (
    "/tmp/pdf-assistant/ats/.analysis_cache.json"
    if os.getenv("VERCEL")
    else "data/ats/.analysis_cache.json"
)


def _cache_key(resume_text: str, job_description: str, company_context: str, llm) -> str:
    model_name = getattr(llm, "model_name", getattr(llm, "model", ""))
    payload = "\x00".join((str(model_name), str(ats_prompt), resume_text, job_description, company_context))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_analysis(cache_key: str) -> ATSAnalysis | None:
    cache_path = os.getenv("ATS_CACHE_PATH", DEFAULT_CACHE_PATH)

    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            cached_data = json.load(cache_file)
        cached_analysis = cached_data.get(cache_key)
        return ATSAnalysis.model_validate(cached_analysis) if cached_analysis else None
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None


def _save_cached_analysis(cache_key: str, analysis: ATSAnalysis) -> None:
    cache_path = os.getenv("ATS_CACHE_PATH", DEFAULT_CACHE_PATH)
    cache_directory = os.path.dirname(cache_path)

    try:
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)

        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cache = json.load(cache_file)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            cache = {}

        cache[cache_key] = analysis.model_dump()

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_directory or ".",
            delete=False
        ) as temporary_file:
            json.dump(cache, temporary_file, ensure_ascii=True, indent=2)
            temporary_path = temporary_file.name

        os.replace(temporary_path, cache_path)
    except OSError:
        # Caching is an optimization; a read-only deployment must still work.
        if "temporary_path" in locals():
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _canonicalize_items(items: list[str]) -> list[str]:
    unique_items = {}
    for item in items:
        normalized_item = item.strip()
        if normalized_item:
            unique_items.setdefault(normalized_item.casefold(), normalized_item)
    return sorted(unique_items.values(), key=str.casefold)


def _normalize_analysis(analysis: ATSAnalysis) -> ATSAnalysis:
    matching_skills = _canonicalize_items(analysis.matching_skills)
    matching_keywords = _canonicalize_items(analysis.matching_keywords)
    matching_skill_keys = {item.casefold() for item in matching_skills}
    matching_keyword_keys = {item.casefold() for item in matching_keywords}

    return analysis.model_copy(update={
        "matching_skills": matching_skills,
        "missing_skills": [
            item for item in _canonicalize_items(analysis.missing_skills)
            if item.casefold() not in matching_skill_keys
        ],
        "matching_keywords": matching_keywords,
        "missing_keywords": [
            item for item in _canonicalize_items(analysis.missing_keywords)
            if item.casefold() not in matching_keyword_keys
        ],
        "strengths": _canonicalize_items(analysis.strengths),
        "recommendations": _canonicalize_items(analysis.recommendations),
    })


def calculate_score(analysis: ATSAnalysis) -> int:
    """
    Calculate the final ATS score deterministically.
    The LLM does not decide the numerical score.
    """

    matching_skills = len(analysis.matching_skills)
    missing_skills = len(analysis.missing_skills)

    matching_keywords = len(analysis.matching_keywords)
    missing_keywords = len(analysis.missing_keywords)

    total_skills = matching_skills + missing_skills
    total_keywords = matching_keywords + missing_keywords

    skill_score = (
        matching_skills / total_skills * 100
        if total_skills > 0
        else 0
    )

    keyword_score = (
        matching_keywords / total_keywords * 100
        if total_keywords > 0
        else 0
    )

    score = (
        skill_score * 0.65 +
        keyword_score * 0.35
    )

    return round(score)


def get_match_level(score: int) -> str:

    if score >= 90:
        return "Excellent Match"

    if score >= 75:
        return "Strong Match"

    if score >= 60:
        return "Moderate Match"

    if score >= 40:
        return "Weak Match"

    return "Poor Match"


def analyze_resume(
    resume_text: str,
    job_description: str,
    llm,
    company_context: str = ""
) -> ATSResult:

    cache_key = _cache_key(
        resume_text,
        job_description,
        company_context,
        llm
    )
    analysis = _load_cached_analysis(cache_key)

    if analysis is None:
        chain = ats_prompt | llm

        response = chain.invoke(
            {
                "resume": resume_text,
                "job_description": job_description,
                "company_context": company_context
            }
        )

        content = response.content

        # Remove markdown code fences if the model adds them.
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]

        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            data = json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON.\n\n"
                f"Model output:\n{content}"
            ) from e

        analysis = ATSAnalysis.model_validate(data)
        analysis = _normalize_analysis(analysis)
        _save_cached_analysis(cache_key, analysis)

    score = calculate_score(analysis)

    match_level = get_match_level(score)

    return ATSResult(
        overall_score=score,
        match_level=match_level,
        analysis=analysis
    )