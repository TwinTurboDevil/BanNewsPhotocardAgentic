# run_agentic_workflow.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)   # noisy CrewAI internal deprecation notices

import os
import csv
import json
import re
import logging
import time
from datetime import datetime
from crewai import Agent, Crew, Process, Task, LLM
from custom_tools import (
    crop_news_photo_llm,
    extract_headline_llm,
    verify_outlet_branding,
    compare_logo_to_reference_style,
    extract_bangla_date,
    search_live_news_free
)

# ==========================================
# CONFIGURATION
# ==========================================
REAL_IMAGE_FOLDER = "G:/News Photocard Research Thesis/Real"   # "actual" will be set to "Real" for these
FAKE_IMAGE_FOLDER = "G:/News Photocard Research Thesis/Fake"   # "actual" will be set to "Fake" for these

OUTPUT_CSV = "G:/News Photocard Research Thesis/Results and Classification/gemini_agentic_verification_results.csv"
LOG_FILE = "G:/News Photocard Research Thesis/Results and Classification/gemini_agentic_run.log"

CSV_FIELDS = ["image_name", "actual", "prediction", "confidence", "verification_case", "reason", "matched_url"]

# Retry settings for degenerate/garbage model output (GPU/VRAM symptom)
MAX_JUDGE_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

# TOGGLE THIS SWITCH FOR YOUR TWO RESEARCH EXPERIMENTS:
# Set False -> Run API-Based Workflow (Gemini)
# Set True  -> Run FULLY LOCAL Open-Source Workflow (Qwen2.5-VL:7b via Ollama)
#              No API-based LLM is called anywhere in this mode -- crop_news_photo_llm
#              and extract_headline_llm in custom_tools.py check PIPELINE_MODE and
#              route to the local Ollama model instead of Gemini.
USE_OPEN_SOURCE = False

os.environ["PIPELINE_MODE"] = "open_source" if USE_OPEN_SOURCE else "api"

# ==========================================
# LOGGING (console + file, so progress survives a crash/interruption)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("agentic_workflow")

if USE_OPEN_SOURCE:
    log.info("Initializing FULLY LOCAL Open-Source Pipeline (Qwen2.5-VL:7b via Ollama)...")
    log.info("No API-based LLM will be called anywhere in this run -- reasoning AND vision are both local.")
    llm_engine = LLM(
        model="ollama/qwen2.5vl:7b",   # vision-capable variant -- NOT plain qwen2.5:7b (text-only)
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
else:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable before running in API mode.")
    log.info("Initializing API-Based LLM Engine (Gemini 3.5 Flash)...")
    llm_engine = LLM(
        model="gemini/gemini-3.5-flash",
        api_key=GEMINI_API_KEY
    )

# ==========================================
# MULTI-AGENT DEFINITIONS
# ==========================================
extraction_agent = Agent(
    role="Visual & Metadata Extraction Specialist",
    goal="Crop news photo area, transcribe the exact Bangla headline, verify outlet branding (logo + website URL), and extract the date.",
    backstory="You are an expert digital forensics technician specializing in visual component extraction.",
    tools=[crop_news_photo_llm, extract_headline_llm, verify_outlet_branding, compare_logo_to_reference_style, extract_bangla_date],
    llm=llm_engine,
    verbose=False
)

retrieval_agent = Agent(
    role="Live News Verification Investigator",
    goal="Search live news archives using free search tool to locate online articles matching headline and date.",
    backstory="You are an investigative journalist skilled at tracking primary news sources across web archives.",
    tools=[search_live_news_free],
    llm=llm_engine,
    verbose=False
)

judge_agent = Agent(
    role="Senior Fact-Checking Magistrate",
    goal="Synthesize extraction and search results to issue final verdict, confidence score, and article URL if Real.",
    backstory="You are a chief editorial magistrate determining whether news photocards are Real or Fake.",
    llm=llm_engine,
    verbose=False
)


def clean_json_output(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _is_degenerate_output(text, min_len=10, repeat_ratio=0.5):
    """
    Detects degenerate model generation -- e.g. '@@@@@@@@@@@@' or 'GGGGGGGG' --
    which on local Ollama setups usually signals a GPU/VRAM problem (KV cache
    overflow, flash-attention issues on older GPUs, or a corrupted model),
    NOT a genuine model judgment. Worth detecting separately so these get
    retried rather than silently recorded as real 'Fake' verdicts.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < min_len:
        return False
    # If a single character makes up most of the output, it's degenerate.
    most_common_count = max(stripped.count(c) for c in set(stripped))
    return (most_common_count / len(stripped)) > repeat_ratio


def _extract_crew_output_text(raw_result):
    """
    str(raw_result) isn't always reliable across CrewAI versions/output modes --
    prefer .raw (the actual final-task text) and fall back progressively.
    """
    output_text = ""
    try:
        output_text = raw_result.raw or ""
    except AttributeError:
        pass
    if not output_text.strip():
        try:
            output_text = raw_result.tasks_output[-1].raw or ""
        except (AttributeError, IndexError):
            pass
    if not output_text.strip():
        output_text = str(raw_result)
    return output_text


def process_single_photocard(image_path, image_name):
    task1 = Task(
        description=(
            f"Analyze photocard at '{image_path}'. Crop the news photo using crop_news_photo_llm, "
            f"transcribe the exact Bangla headline text using extract_headline_llm (this tool "
            f"already excludes ad/sponsor banners and sub-paragraph body text -- it returns only "
            f"the headline, plus an optional bracketed note of what it excluded, for your reference), "
            f"verify the outlet's branding using verify_outlet_branding (this checks the logo "
            f"pixel match AND vision-LLM recognition AND any visible website URL against known real "
            f"domains -- an outlet may show its icon-only logo with no name, or its full text logo, "
            f"or reveal its domain in a footer/watermark instead of a clear logo, so check all of "
            f"these rather than relying on one signal). Then, using the outlet name that "
            f"verify_outlet_branding recognized (or your own reading of the photocard if it found "
            f"none), call compare_logo_to_reference_style with that outlet name -- this does a "
            f"direct visual style comparison against your collected reference logos for that "
            f"specific outlet, and is generally MORE reliable than the SSIM score alone, so weigh "
            f"its result more heavily. Finally extract the date via extract_bangla_date, and note "
            f"the news outlet's name/brand as visibly shown on the photocard."
        ),
        expected_output=(
            "A summary including: (1) the cropped photo path, (2) the EXACT extracted Bangla "
            "headline text verbatim (this is critical -- do not translate or paraphrase it), "
            "(3) the outlet branding verification result from verify_outlet_branding (logo pixel "
            "match + VLM recognition + website URL match), (4) the direct style-comparison result "
            "from compare_logo_to_reference_style (matches_reference_style, confidence, reasoning), "
            "(5) the extracted date -- either an ISO date (YYYY-MM-DD), a 'No Absolute Date -- "
            "relative time reference found' note (e.g. photocard says 'today' with no calendar "
            "date -- this is NOT a failure, proceed without a date), or 'Date Not Extracted', "
            "and (6) the outlet name/brand as visibly shown on the photocard."
        ),
        agent=extraction_agent
    )

    task2 = Task(
        description=(
            "Using the outputs from the previous task, call search_live_news_free with: "
            "headline = the EXACT Bangla headline text, copied VERBATIM in Bangla script from the "
            "previous task's output. This is the single most important input. Do NOT translate it, "
            "do NOT summarise it, do NOT substitute an English description of it, and do NOT pass "
            "any placeholder, bracketed note, or the image filename. If the value you are about to "
            "pass does not contain actual Bangla characters, you have the wrong value -- go back and "
            "find the real headline text. "
            "outlet_domain = the domain confirmed by verify_outlet_branding if it found one. "
            "This dataset only contains photocards from these ten Bangladeshi outlets, so the domain "
            "MUST be one of: prothomalo.com, bangla.thedailystar.net, jugantor.com, kalerkantho.com, "
            "samakal.com, somoynews.tv, channel24bd.tv, jamuna.tv, itvbd.com, rtvonline.com. "
            "NEVER pass a non-Bangladeshi domain (e.g. abcnews.com, cnn.com) -- if the outlet you see "
            "does not map to one of the ten above, pass '' instead of guessing; "
            "target_date = the ISO date extracted in the previous task, or '' if no absolute date "
            "was found (including the 'relative time reference' case -- pass '' for that too, "
            "since there is no calendar date to search with). "
            "The tool checks the CLAIMED outlet first (site-restricted search, then Wayback archive), "
            "and only falls back to a broad, unrestricted web search if the claimed outlet shows "
            "nothing -- report back exactly which case the tool returned: CONFIRMED ON CLAIMED "
            "OUTLET, WEAK SUPPORTING EVIDENCE (Wayback only), CASE A (possible misattribution -- "
            "found on a different outlet), or CASE B (no match anywhere)."
        ),
        expected_output=(
            "Which of the four cases the tool returned (confirmed on claimed outlet / weak Wayback "
            "evidence / Case A possible misattribution / Case B no match anywhere), the candidate "
            "article(s) or archived snapshot URL found (if any), and which other outlet the story "
            "was found on instead, if Case A applies. Judge whether a result MATCHES based on the "
            "headline text, outlet, and date only -- do NOT reject an otherwise-matching article "
            "because its picture differs from the photocard's picture, since outlets commonly use "
            "a different graphic on the social photocard than on the article page."
        ),
        agent=retrieval_agent,
        context=[task1],
    )

    task3 = Task(
        description="""
        Synthesize all findings and issue a verdict. Important judgment rules:
        - "CONFIRMED ON CLAIMED OUTLET" from the search tool is strong evidence for Real. If an
          absolute date was extracted, it should also be consistent with the article found. If NO
          absolute date was available (Date Not Extracted, or a relative time reference like
          "today" with no calendar date), do NOT penalize this -- judge on headline + outlet match
          alone; a missing date is not the same as a conflicting date.
        - "WEAK SUPPORTING EVIDENCE" (Wayback-only) confirms the outlet's site existed around
          that time, not that this specific headline was published -- do not treat it as
          equivalent to a direct search match. Weigh it alongside the logo/website branding signals.
        - "CASE A" (possible misattribution) means the story was found, but on a DIFFERENT
          outlet than the one claimed on the photocard. This should be judged Fake for the
          claimed outlet, with the rationale explicitly noting this is likely source
          impersonation of a genuine story, not fabricated content.
        - "CASE B" (no match anywhere) means nothing was found on the claimed outlet or
          elsewhere on the open web. This should be judged Fake, with the rationale noting
          this looks like fabricated content rather than misattribution.
        - CRITICAL EXCEPTION: if the search tool returned "SEARCH NOT PERFORMED -- INVALID QUERY",
          then NO search actually happened. This is a pipeline failure, not evidence about the
          photocard. Do NOT treat it as Case B and do NOT conclude Fake on that basis. Set
          verification_case to "search_failed" and base your verdict only on the branding/logo
          evidence, with reduced confidence.
        - Also note: the search backend indexes Bangla-language news imperfectly. A "no match"
          result is weaker evidence than a positive match -- weigh it against the branding
          signals rather than treating absence of search results as conclusive proof of fakery,
          especially when the logo and outlet branding verify as authentic.
        - IMAGE MATCHING -- IMPORTANT: do NOT require the photocard's picture to match the
          article's picture. News outlets routinely use a DIFFERENT graphic on their social
          media photocard than on the article page itself (e.g. a plain background with just
          an official seal/logo on the photocard, versus a photo or illustration on the
          article). A photocard is authentic if the HEADLINE, the OUTLET, and the DATE match
          the published article -- the accompanying image only needs to be topically related
          (same subject/organisation/event), not identical or even visually similar. Only
          treat the image as evidence of fakery if it clearly CONTRADICTS the story (e.g. an
          entirely unrelated subject, or a picture of a different person than the one the
          headline is about). Never predict Fake solely because the images differ.
        - Weigh the outlet branding verification (logo match AND/OR website URL match) alongside
          the search evidence -- a photocard can fail on branding even if a real story exists
          elsewhere, or pass branding but still fail retrieval.

        CONFIDENCE CALIBRATION -- read carefully, this is not a lookup table:
        Do NOT pick confidence from a mental list of "typical" numbers (0.85, 0.95, 0.99, 0.5).
        Do NOT let verification_case alone decide confidence -- two cases with the SAME
        verification_case can warrant very different confidence depending on how many other
        signals (branding, logo, domain, language quality) agree or disagree with it. Think
        of confidence as a genuine, continuous estimate of "how likely am I to be right about
        THIS specific image," reasoned from scratch each time -- it will often be an unremarkable,
        non-round number like 0.62, 0.71, or 0.88, not a tidy default.

        Some guiding intuition, NOT fixed rules -- the actual number should come from your
        reasoning about this specific case, and can fall anywhere the evidence genuinely supports:
        - When multiple independent signals agree strongly (e.g. a direct article match on the
          claimed outlet AND clean branding, or a clear branding forgery AND no article found
          anywhere), that convergence is strong evidence -- confidence can reasonably be quite
          high, but only because of the AGREEMENT, not because the verdict happens to be Fake
          or Real.
        - When your only basis is a single weak or ambiguous signal -- e.g. nothing found by
          search but branding looks unremarkable either way, or a Wayback-only hit with no
          direct headline match -- your genuine uncertainty should show. It is entirely normal,
          and expected, for a confidence value to sit close to the 0.5 boundary in these cases,
          on EITHER side of it, rather than being pushed toward a "safe-looking" higher number
          just because you must still pick a definitive verdict. A close call that turns out
          wrong is a normal, expected outcome of genuine uncertainty -- do not distort your
          confidence just to make the verdict look decisive.
        - Do not treat "the verdict is Fake" as inherently deserving lower or higher confidence
          than "the verdict is Real" -- confidence tracks evidence strength, not which label
          you landed on.

        Set the "verification_case" field based on which of these applied:
        "confirmed", "weak_wayback_only", "possible_misattribution", "no_match_anywhere", "search_failed".

        Output strictly in JSON format with no extra commentary:
        {
          "prediction": "Real" or "Fake",
          "confidence": a number between 0.0 and 1.0, reasoned freshly per the CONFIDENCE CALIBRATION guidance above -- not selected from a fixed set of values,
          "verification_case": "confirmed" or "weak_wayback_only" or "possible_misattribution" or "no_match_anywhere" or "search_failed",
          "reason": "1-2 sentences citing visual, branding, and retrieval evidence, including which case applied and why that specific confidence value fits the strength of evidence.",
          "matched_url": "URL of the article found (on the claimed outlet if Real, or on the other outlet if Case A), otherwise empty string ''"
        }
        """,
        expected_output="JSON formatted verdict.",
        agent=judge_agent,
        context=[task1, task2],
    )

    crew = Crew(
        agents=[extraction_agent, retrieval_agent, judge_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=False
    )

    # Retry loop: a degenerate/garbage generation (e.g. '@@@@@@@@') is a
    # transient GPU/VRAM symptom on local Ollama setups, not a real verdict --
    # retry rather than permanently recording a false 'Fake'.
    output_text = ""
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        raw_result = crew.kickoff()
        output_text = _extract_crew_output_text(raw_result)

        if _is_degenerate_output(output_text):
            snippet = output_text.strip()[:80]
            log.warning(
                f"[{image_name}] Degenerate model output on attempt {attempt}/{MAX_JUDGE_ATTEMPTS}: {snippet!r} "
                f"-- this usually means a GPU/VRAM problem (try a smaller model, lower num_ctx, "
                f"OLLAMA_FLASH_ATTENTION=0, or restarting Ollama)."
            )
            if attempt < MAX_JUDGE_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
        break

    cleaned_json_str = clean_json_output(output_text)

    try:
        parsed = json.loads(cleaned_json_str)
        prediction = parsed.get("prediction", "Fake").capitalize()
        confidence = float(parsed.get("confidence", 0.5))
        verification_case = parsed.get("verification_case", "unknown")
        reason = parsed.get("reason", "No reason provided.")
        matched_url = parsed.get("matched_url", "")
    except Exception as e:
        prediction = "Fake"
        confidence = 0.5
        # Distinguish a GPU/infrastructure failure from a genuine formatting
        # failure -- these should NOT be lumped together in your error analysis.
        raw_snippet = output_text.strip()[:300] if output_text.strip() else "(completely empty output)"
        if _is_degenerate_output(output_text):
            verification_case = "degenerate_output_gpu_error"
            reason = (
                f"Degenerate model output after {MAX_JUDGE_ATTEMPTS} attempts (likely GPU/VRAM issue, "
                f"NOT a real verdict -- exclude from metrics): {raw_snippet!r}"
            )
        else:
            verification_case = "parse_error"
            reason = f"Parsing error: {str(e)} | Raw judge output was: {raw_snippet!r}"
        matched_url = ""
        log.warning(f"[{image_name}] Judge output unusable ({verification_case}). Raw: {raw_snippet!r}")

    return {
        "image_name": image_name,
        "prediction": prediction,
        "confidence": confidence,
        "verification_case": verification_case,
        "reason": reason,
        "matched_url": matched_url
    }


def _load_already_processed(output_csv):
    """Reads any existing output CSV so an interrupted run can resume without
    reprocessing (and re-paying API/compute cost for) images already done."""
    done = set()
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_name"):
                    done.add(row["image_name"])
    return done


def _collect_jobs():
    """Builds the full (image_path, image_name, actual_label) job list from both folders."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    jobs = []
    for folder, actual_label in [(REAL_IMAGE_FOLDER, "Real"), (FAKE_IMAGE_FOLDER, "Fake")]:
        if not os.path.isdir(folder):
            log.warning(f"Folder not found, skipping: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
        log.info(f"Found {len(files)} images in {folder} (actual={actual_label})")
        for f in files:
            jobs.append((os.path.join(folder, f), f, actual_label))
    return jobs


def run_batch_evaluation():
    jobs = _collect_jobs()
    if not jobs:
        log.error("No images found in either REAL_IMAGE_FOLDER or FAKE_IMAGE_FOLDER. Check the paths in config.")
        return

    already_done = _load_already_processed(OUTPUT_CSV)
    if already_done:
        log.info(f"Resuming: {len(already_done)} images already in {OUTPUT_CSV} will be skipped.")

    remaining = [(p, n, a) for (p, n, a) in jobs if n not in already_done]
    log.info(f"{len(remaining)} of {len(jobs)} images remaining to process.")

    if not remaining:
        log.info("Nothing left to do -- all images already processed.")
        return

    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

    # Open in append mode and write each row immediately -- this is the
    # "save progress in real time" behavior: if the process is interrupted
    # (crash, Ctrl+C, power loss), everything completed so far is already
    # safely on disk in OUTPUT_CSV and _load_already_processed() will skip
    # it automatically on the next run.
    with open(OUTPUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for idx, (img_path, img_name, actual_label) in enumerate(remaining, start=1):
            log.info(f"[{idx}/{len(remaining)}] Processing {img_name} (actual={actual_label})...")
            try:
                res = process_single_photocard(img_path, img_name)
            except Exception as e:
                log.error(f"[{img_name}] Unhandled error, recording as parse_error/Fake: {e}")
                res = {
                    "image_name": img_name,
                    "prediction": "Fake",
                    "confidence": 0.0,
                    "verification_case": "unhandled_error",
                    "reason": f"Unhandled exception: {e}",
                    "matched_url": "",
                }

            row = {"image_name": res["image_name"], "actual": actual_label}
            row.update({k: res[k] for k in ["prediction", "confidence", "verification_case", "reason", "matched_url"]})

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())  # force to disk immediately, don't rely on OS buffering

            log.info(
                f"[{img_name}] actual={actual_label} -> prediction={res['prediction']} "
                f"(conf={res['confidence']}, case={res['verification_case']})"
            )
            if res.get("matched_url"):
                log.info(f"  matched_url: {res['matched_url']}")

    log.info(f"Execution complete. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    start = datetime.now()
    log.info(f"=== Run started {start.isoformat()} | PIPELINE_MODE={os.environ['PIPELINE_MODE']} ===")
    run_batch_evaluation()
    log.info(f"=== Run finished, elapsed: {datetime.now() - start} ===")