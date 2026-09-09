import json
import os
import re
import csv
import logging
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Paste your Gemini API key inside the quotes below:
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

REAL_IMAGE_FOLDER = "E:/News Photocard Research Thesis/Real"   # "actual" will be set to "Real" for these
FAKE_IMAGE_FOLDER = "E:/News Photocard Research Thesis/Fake"   # "actual" will be set to "Fake" for these

OUTPUT_CSV = "E:/News Photocard Research Thesis/gemini_baseline_result.csv"
LOG_FILE = "E:/News Photocard Research Thesis/gemini_baseline_run.log"

MODEL_NAME = "gemini-3.5-flash"  # Lightweight vision LLM for fast baseline evaluation

CSV_FIELDS = ["image_name", "actual", "prediction", "confidence", "reason"]

SYSTEM_PROMPT = """
You are a meticulous fact-checker examining a Bangla news photocard shared on social media. A photocard may be authentic (genuinely published by the outlet shown) or fabricated (falsely attributed, edited, or invented).

You have NO internet access and cannot search for anything. Base your judgment only on:
(a) what is visible in the image itself, and
(b) your own general knowledge of relevant events, as known to you.

Examine the photocard across these six dimensions. Do not assume any specific manipulation pattern in advance — evaluate each dimension independently based on what you actually observe in THIS image:

1. Outlet identity: Does the name, logo, and branding shown match a real, known news outlet exactly? Note any deviation in spelling, characters, or logo design, however subtle — without assuming which outlet is involved.
2. Domain/URL: If a website address or handle is visible, does it plausibly match that outlet's real domain/handle?
3. Visual/template consistency: Are fonts, layout, spacing, colors, and any template elements consistent with a professional news outlet's normal publishing style? Note any signs of digital editing (blurring, mismatched fonts, color patches, compression artifacts around text).
4. Language quality: Is the Bangla text written in standard, professional editorial style? Note broken conjuncts (যুক্তাক্ষর), unnatural phrasing, informal slang, or a clickbait tone atypical of professional news writing.
5. Factual/content plausibility: Based only on your own knowledge (not searching), does the claimed event, date, and people involved seem consistent with what you actually know? Flag anything that contradicts known facts, seems anachronistic, or describes a notable event you have no memory of despite its apparent scale.
6. Internal consistency: Do the headline, image, and any visible date/context align with each other without contradiction?

You must give a definitive verdict — "Real" or "Fake" — even under uncertainty. Choose the more likely option based on the balance of evidence. Do not respond with "Unknown" or "Uncertain."

CONFIDENCE CALIBRATION -- read carefully, this is not a lookup table:
You have NO way to verify this externally -- no search, no reference database. Do NOT pick
confidence from a mental list of "typical" numbers (0.85, 0.95, 0.5) and do NOT let your
verdict (Real vs Fake) alone decide how confident you are -- reason about THIS specific
image's evidence each time, from scratch.

Some guiding intuition, NOT fixed rules -- the number should come from genuine reasoning
about how many of the 6 dimensions show clear, specific signal:
- If several dimensions independently point the same direction with concrete, specific
  detail you can name (an exact misspelling, a specific broken conjunct, a professional
  template you recognize clearly) that convergence is real evidence and confidence can
  reasonably be high -- but only because of that convergence, not because of which label
  you chose.
- If your only basis is vague or singular -- e.g. you simply do not personally recall the
  event, or one minor stylistic quirk you are not fully sure about -- your genuine
  uncertainty should show in the number. It is normal and expected for confidence to sit
  close to the 0.5 boundary in these cases, on EITHER side, rather than defaulting to a
  "safe-looking" higher value just because a definitive verdict is required. Not recognizing
  an event is NOT proof it is fabricated -- your knowledge has a cutoff and cannot cover
  recent real events, so let that uncertainty be reflected honestly rather than overridden.
- A confidence value close to 0.5 that turns out wrong is a normal, expected outcome of
  genuine uncertainty -- do not distort the number just to make the verdict look decisive.

Respond strictly in this JSON format, with no extra commentary, markdown, or code fences:
{
  "prediction": "Real" or "Fake",
  "confidence": a number between 0.0 and 1.0, reasoned freshly per the CONFIDENCE CALIBRATION guidance above -- not selected from a fixed set of values,
  "reason": "1-2 sentences citing which of the 6 dimensions most influenced this verdict and why that specific confidence value fits the strength of evidence"
}
"""

# ==========================================
# LOGGING (console + file)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("gemini_baseline")


def clean_json_response(raw_text):
  """Extracts valid JSON even if the model wraps output in markdown code blocks."""
  raw_text = raw_text.strip()
  # Remove markdown code blocks if present
  raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
  raw_text = re.sub(r"\s*```$", "", raw_text)

  match = re.search(r"\{.*\}", raw_text, re.DOTALL)
  if match:
    return match.group(0)
  return raw_text


def _collect_jobs():
  """Builds the (image_path, image_name, actual_label) job list from both folders."""
  valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
  jobs = []
  for folder, actual_label in [(REAL_IMAGE_FOLDER, "Real"), (FAKE_IMAGE_FOLDER, "Fake")]:
    if not os.path.isdir(folder):
      log.warning(f"Folder not found, skipping: {folder}")
      continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
    log.info(f"Found {len(files)} images in {folder} (actual={actual_label})")
    for f in files:
      jobs.append((os.path.join(folder, f), f, actual_label))
  return jobs


def _load_already_processed(output_csv):
  """Reads any existing output CSV so an interrupted run can resume without
  reprocessing images already done. This is the checkpoint mechanism."""
  done = set()
  if os.path.exists(output_csv):
    with open(output_csv, "r", encoding="utf-8-sig", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        if row.get("image_name"):
          done.add(row["image_name"])
  return done


def process_single_photocard(client, img_path, img_name, actual_label):
  """Runs one image through the API baseline model. Returns a result dict.
  This is unchanged logic from the original script -- just pulled into its
  own function so the main loop can write each result immediately."""
  try:
    pil_image = Image.open(img_path)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_image, SYSTEM_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,  # Zero-shot deterministic baseline evaluation
            response_mime_type="application/json",
        ),
    )

    raw_output = response.text
    cleaned_json_str = clean_json_response(raw_output)

    try:
      parsed = json.loads(cleaned_json_str)
      prediction = parsed.get("prediction", "Fake").capitalize()
      confidence = float(parsed.get("confidence", 0.5))
      reason = parsed.get("reason", "No reason provided.")
    except Exception:
      # Fallback regex extraction if strict JSON syntax parsing breaks
      pred_match = re.search(r'"prediction"\s*:\s*"([^"]+)"', raw_output)
      conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_output)
      reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_output)

      prediction = pred_match.group(1).capitalize() if pred_match else "Fake"
      confidence = float(conf_match.group(1)) if conf_match else 0.0
      reason = (
          reason_match.group(1)
          if reason_match
          else "Extracted via regex fallback due to malformed JSON."
      )

    if prediction not in ["Real", "Fake"]:
      prediction = "Fake"

  except Exception as e:
    prediction = "Error"
    confidence = 0.0
    reason = f"Execution failed: {str(e)}"

  return {
      "image_name": img_name,
      "actual": actual_label,
      "prediction": prediction,
      "confidence": confidence,
      "reason": reason,
  }


def run_api_baseline():
  # Priority: 1. Hardcoded API key in script -> 2. Environment variable
  api_key = (
      GEMINI_API_KEY
      if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
      else os.getenv("GEMINI_API_KEY")
  )

  if not api_key:
    log.error("Gemini API key is missing.")
    log.error(
        "Please paste your API key in the GEMINI_API_KEY variable at the top of"
        " this script, or set the GEMINI_API_KEY environment variable."
    )
    return

  client = genai.Client(api_key=api_key)

  jobs = _collect_jobs()

  if not jobs:
    log.error("No images found in either REAL_IMAGE_FOLDER or FAKE_IMAGE_FOLDER. Please check the paths.")
    return

  already_done = _load_already_processed(OUTPUT_CSV)
  if already_done:
    log.info(f"Resuming: {len(already_done)} images already in {OUTPUT_CSV} will be skipped.")

  remaining = [job for job in jobs if job[1] not in already_done]
  log.info(f"{len(remaining)} of {len(jobs)} images remaining to process.")

  if not remaining:
    log.info("Nothing left to do -- all images already processed.")
    return

  log.info(
      f"Starting API-based Vision LLM ({MODEL_NAME}) Baseline on {len(remaining)} images..."
  )

  real_count = 0
  fake_count = 0
  error_count = 0

  file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

  # Open in append mode and write each row immediately -- this is the
  # checkpoint: if the process is interrupted (crash, Ctrl+C, rate limit),
  # everything completed so far is already safely on disk, and re-running
  # this script will automatically skip it via _load_already_processed().
  with open(OUTPUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if not file_exists:
      writer.writeheader()
      f.flush()

    for img_path, img_name, actual_label in tqdm(remaining, desc="Processing Photocards"):
      res = process_single_photocard(client, img_path, img_name, actual_label)

      if res["prediction"] == "Real":
        real_count += 1
      elif res["prediction"] == "Fake":
        fake_count += 1
      elif res["prediction"] == "Error":
        error_count += 1

      log.info(
          f"[Image: {img_name}] actual={actual_label} --> "
          f"Prediction: {res['prediction']} (Conf: {res['confidence']})"
      )

      writer.writerow(res)
      f.flush()
      os.fsync(f.fileno())  # force to disk immediately, don't rely on OS buffering

  log.info("\n" + "=" * 50)
  log.info("API BASELINE EVALUATION SUMMARY (this run)")
  log.info("=" * 50)
  log.info(f"Images processed this run : {len(remaining)}")
  log.info(f"Predicted REAL             : {real_count}")
  log.info(f"Predicted FAKE             : {fake_count}")
  log.info(f"Execution Errors           : {error_count}")
  log.info(f"Results saved to           : {OUTPUT_CSV}")
  log.info("=" * 50)


if __name__ == "__main__":
  run_api_baseline()