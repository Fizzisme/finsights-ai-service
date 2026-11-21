"""Draft the questions of the retrieval eval v2 from sampled articles.

Each case starts from one "seed" article of a month: an LLM writes a question of the wanted
category that the article answers. The output is a draft (`relevant`/`irrelevant` empty);
`evals/builders/judge.py` pools what the retrievers return and labels it.

Questions written from one article tend to reuse its words, which makes search look easier
than it is; the `paraphrase` and `english` prompts forbid that, and the draft is read by hand
before judging.

    uv run --env-file .env python -m evals.builders.generate_cases
"""

import asyncio
import json
import random
import re
import selectors
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Articles collected after this id are outside the eval, so the corpus does not move when the
# news pipeline adds more (ids follow collection order, not publication date).
MAX_ARTICLE_ID = 75560
MONTHS = [
    (date(2026, 9, 1), date(2026, 9, 30)),
    (date(2026, 10, 1), date(2026, 10, 31)),
    (date(2026, 11, 1), date(2026, 11, 20)),
]
PER_MONTH = {
    "lexical": 6,
    "multiword": 5,
    "paraphrase": 6,
    "english": 4,
    "detail": 6,
    "date": 3,
}
MIN_BODY = 1500
OUT = Path(__file__).resolve().parents[1] / "datasets" / "news_topics_v2.draft.jsonl"

COMMON = """Bạn viết câu hỏi cho bộ đánh giá tìm kiếm tin tài chính tiếng Việt.
Người dùng hỏi một trợ lý; trợ lý tìm trong kho tin rồi trả lời. Dưới đây là một bài báo.
Viết MỘT câu hỏi tự nhiên như người dùng thật, mà bài này trả lời được.
{rule}
Ngoài câu hỏi, viết `query`: 1-4 từ khoá tiếng Việt mà trợ lý sẽ gõ vào ô tìm kiếm cho câu hỏi.
Chỉ trả về JSON: {schema}

Ngày đăng: {published}
Tiêu đề: {title}
Tóm tắt: {summary}
Nội dung:
{body}"""

RULES = {
    "lexical": (
        "Câu hỏi về CHỦ ĐỀ chính của bài, đủ chung để các bài khác cùng chủ đề cũng trả lời "
        "được, và nhắc đúng cụm từ chính như bài viết (vd: 'Giá vàng trong tháng {month} "
        "biến động thế nào?'). Nhắc tháng {month}."
    ),
    "multiword": (
        "Câu hỏi ghép HAI ý của bài (vd: một công ty + một chỉ tiêu: 'Lợi nhuận quý 3 của "
        "Hòa Phát thế nào?'), hai cụm đó không đứng cạnh nhau trong tiêu đề. Nhắc tháng "
        "{month} nếu tự nhiên."
    ),
    "paraphrase": (
        "Câu hỏi về chủ đề chính của bài nhưng KHÔNG dùng lại các từ ngữ chính của tiêu đề và "
        "tóm tắt: nói cùng ý bằng từ khác, như người dùng không biết cách báo viết (vd: bài "
        "'khối ngoại bán ròng' -> 'Nhà đầu tư nước ngoài có đang rút vốn không?')."
    ),
    "english": (
        "Câu hỏi bằng TIẾNG ANH về chủ đề chính của bài, như một người nước ngoài hỏi (vd: "
        "'How did gold prices move in {month_en}?'). `query` vẫn là từ khoá tiếng Việt."
    ),
    "detail": (
        "Câu hỏi về MỘT dữ kiện cụ thể (con số, tên, kết quả) chỉ có trong phần nội dung, "
        "KHÔNG có trong tiêu đề hay tóm tắt. Câu hỏi phải tự đủ nghĩa (nhắc tên công ty/cơ "
        "quan). Thêm `evidence`: MỘT câu hoặc đoạn câu CHÉP NGUYÊN VĂN từ nội dung (10-30 từ) "
        "chứa câu trả lời."
    ),
    "date": (
        "Câu hỏi về chủ đề chính của bài nhưng giới hạn trong ngày {day} (vd: 'Ngày {day} "
        "có tin gì về lãi suất ngân hàng?'). Phải nhắc ngày {day}."
    ),
}
SCHEMA = '{"question": "...", "query": "..."}'
DETAIL_SCHEMA = '{"question": "...", "query": "...", "evidence": "..."}'
MONTH_EN = ["", "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December"]  # fmt: skip


def _squash(text: str) -> str:
    return " ".join(text.split())


def vn_date(published: datetime) -> date:
    return (published + timedelta(hours=7)).date()


async def candidates(start: date, end: date) -> list[dict[str, Any]]:
    """Published articles of the period with a body long enough to ask about."""
    from app.storage.mysql.engine import fetch_all
    from app.tools.sql.search_news import vn_day_start_utc

    rows: list[dict[str, Any]] = []
    while True:
        page = await fetch_all(
            "SELECT a.id, a.article_uid, a.title, a.summary, a.published_at "
            "FROM news_articles a JOIN news_article_contents c ON c.article_id = a.id "
            "WHERE a.article_status = 'published' AND a.id > :last AND a.id <= :max_id "
            "AND a.published_at >= :start AND a.published_at < :end "
            "AND CHAR_LENGTH(c.body_plaintext) >= :min_body ORDER BY a.id LIMIT 500",
            {
                "last": rows[-1]["id"] if rows else 0,
                "max_id": MAX_ARTICLE_ID,
                "start": vn_day_start_utc(start),
                "end": vn_day_start_utc(end + timedelta(days=1)),
                "min_body": MIN_BODY,
            },
        )
        rows.extend(page)
        if len(page) < 500:
            break
    return rows


async def body(article_id: int) -> str:
    from app.storage.mysql.engine import fetch_all

    rows = await fetch_all(
        "SELECT body_plaintext FROM news_article_contents WHERE article_id = :id",
        {"id": article_id},
    )
    return rows[0]["body_plaintext"] if rows else ""


def _json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def draft(
    model: Any, category: str, seed: dict[str, Any], start: date, end: date
) -> dict[str, Any] | None:
    """One case from `seed`, or None when the answer is unusable."""
    text = await body(seed["id"])
    day = vn_date(seed["published_at"])
    month = f"{start.month}/{start.year}"
    rule = RULES[category].format(
        month=month,
        month_en=f"{MONTH_EN[start.month]} {start.year}",
        day=f"{day.day}/{day.month}/{day.year}",
    )
    prompt = COMMON.format(
        rule=rule,
        schema=DETAIL_SCHEMA if category == "detail" else SCHEMA,
        published=day.isoformat(),
        title=seed["title"],
        summary=seed["summary"] or "",
        body=text[:6000],
    )
    reply = await model.ainvoke(prompt)
    raw = _json(reply.content if isinstance(reply.content, str) else str(reply.content))
    if not raw or not raw.get("question") or not raw.get("query"):
        return None
    case: dict[str, Any] = {
        "category": category,
        "question": raw["question"].strip(),
        "query": raw["query"].strip(),
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "seed": seed["article_uid"],
        "seed_title": seed["title"],
        "relevant": [],
        "irrelevant": [],
    }
    if category == "date":
        # The day of the article, one day either side: the question names the day, the
        # period is what an agent would filter on.
        case["date_from"] = max(start, day - timedelta(days=1)).isoformat()
        case["date_to"] = min(end, day + timedelta(days=1)).isoformat()
    if category == "detail":
        evidence = _squash(str(raw.get("evidence", "")))
        visible = _squash(f"{seed['title']} {seed['summary'] or ''}")
        # Must be copied from the body, and not answerable from title + summary.
        if not evidence or evidence not in _squash(text) or evidence in visible:
            return None
        case["evidence"] = [evidence]
    return case


async def main() -> None:
    from app.integrations.llm.factory import get_chat_model

    model = get_chat_model("deepseek", "strong")
    rng = random.Random(2026)
    limit = asyncio.Semaphore(6)
    cases: list[dict[str, Any]] = []
    for start, end in MONTHS:
        pool = await candidates(start, end)
        rng.shuffle(pool)
        print(f"{start:%Y-%m}: {len(pool)} candidate articles", file=sys.stderr)
        seeds = iter(pool)
        for category, wanted in PER_MONTH.items():
            made: list[dict[str, Any]] = []
            while len(made) < wanted:
                # Oversample: some drafts fail the checks (mostly detail evidence).
                batch = [next(seeds) for _ in range(wanted - len(made) + 1)]

                async def one(
                    seed: dict[str, Any],
                    category: str = category,
                    start: date = start,
                    end: date = end,
                ) -> Any:
                    async with limit:
                        try:
                            return await draft(model, category, seed, start, end)
                        except Exception as exc:  # noqa: BLE001 - a draft may fail
                            print(f"  {category}: {exc}", file=sys.stderr)
                            return None

                results = await asyncio.gather(*(one(s) for s in batch))
                made.extend(r for r in results if r)
            for n, case in enumerate(made[:wanted], start=1):
                cases.append({"id": f"{start:%m}-{category}-{n}", **case})
            print(f"  {category}: {wanted}", file=sys.stderr)
    with OUT.open("w", encoding="utf-8") as out:
        out.write(json.dumps({"max_article_id": MAX_ARTICLE_ID}) + "\n")
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
    )
