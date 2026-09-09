# custom_tools.py
import os
import re
import cv2
import json
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import easyocr
from crewai.tools import tool

# Initialize GPU-accelerated EasyOCR for Bangla & English
reader = easyocr.Reader(['bn', 'en'], gpu=True)

# ==========================================
# VISION BACKEND SWITCH
# ==========================================
# PIPELINE_MODE is set by agentic_workflow.py from its USE_OPEN_SOURCE toggle.
# "open_source" -> ALL vision calls go to a local Ollama model (Qwen2.5-VL:7b).
#                  No network call to any proprietary/API-based LLM happens.
# "api"         -> vision calls go to Gemini's API.
# Checked at call time (not import time) so ordering of imports never matters.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")


def _is_open_source_mode() -> bool:
    return os.environ.get("PIPELINE_MODE", "api") == "open_source"


def _pil_to_base64(pil_img: Image.Image) -> str:
    """
    Encodes a PIL image to base64 for sending to the vision model.

    Layered fallback: a Pillow 11.3.0 build on Kaggle was found to crash on
    JPEG re-encoding with 'function takes at most 16 arguments (17 given)'.
    Not observed on this local setup, but kept here defensively so this
    function behaves identically across both environments.
    """
    rgb_img = pil_img.convert("RGB")

    try:
        buf = BytesIO()
        rgb_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[_pil_to_base64] JPEG encode failed ({e}), falling back to PNG.")

    try:
        buf = BytesIO()
        rgb_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[_pil_to_base64] PNG encode also failed ({e}), falling back to raw file bytes.")

    source_path = getattr(pil_img, "filename", None)
    if source_path and os.path.exists(source_path):
        with open(source_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    raise RuntimeError(
        "Could not encode image to base64: JPEG and PNG both failed, and no "
        "original file path was available as a last-resort fallback."
    )


def _call_ollama_vision_multi(pil_images: list, prompt: str) -> dict:
    """
    Calls a LOCAL Ollama-hosted vision model (default: qwen2.5vl:7b) with one
    or more images in a single request. Requires `ollama pull qwen2.5vl:7b`
    and the Ollama server running locally. No data leaves the machine -- no
    API key, no external network call.
    """
    payload = {
        "model": OLLAMA_VISION_MODEL,
        "prompt": prompt,
        "images": [_pil_to_base64(img) for img in pil_images],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    resp.raise_for_status()
    return json.loads(resp.json().get("response", "{}"))


def _call_gemini_vision_multi(pil_images: list, prompt: str) -> dict:
    """Calls Gemini's API with one or more images. Only used when PIPELINE_MODE == 'api'."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable missing.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=list(pil_images) + [prompt],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return json.loads(response.text)


def _call_vision_model_multi(pil_images: list, prompt: str) -> dict:
    """Routes a multi-image call to the local or API vision backend based on PIPELINE_MODE."""
    if _is_open_source_mode():
        return _call_ollama_vision_multi(pil_images, prompt)
    return _call_gemini_vision_multi(pil_images, prompt)


def _call_vision_model(pil_img: Image.Image, prompt: str) -> dict:
    """Single-image convenience wrapper around _call_vision_model_multi."""
    return _call_vision_model_multi([pil_img], prompt)


# ==========================================
# KNOWN OUTLET DOMAINS
# ==========================================
# Used to check whether a website URL/domain visible on the photocard matches
# a real outlet's actual domain. All 10 real outlets in your dataset
# (BanglaFakeCard/Real/) are filled in below, each verified against a live
# source. Fake-outlet domains are intentionally NOT included here, per your
# instruction -- this dict only needs to answer "does this match a REAL
# outlet's real domain," so parody/fake domains have no reason to be in it.
#
# NOTE on "daily star": headlines in your dataset are Bangla, so Task 2 maps
# "Daily Star" -> bangla.thedailystar.net (the separate Bangla-language site),
# not thedailystar.net (the English site) -- both are listed below since a
# photocard could in principle show either, but the Bangla one is the default.
#
# NOTE on "independent tv": confirmed as itvbd.com per direct user confirmation
# (search sources were split between this and independent24.com -- itvbd.com
# is the one that's actually correct).
KNOWN_OUTLET_DOMAINS = {
    "prothom alo": "prothomalo.com",
    "daily star bangla": "bangla.thedailystar.net",
    "daily star": "bangla.thedailystar.net",   # default to Bangla site -- see NOTE above
    "jugantor": "jugantor.com",
    "kaler kontho": "kalerkantho.com",
    "kaler kantho": "kalerkantho.com",         # alternate transliteration
    "samakal": "samakal.com",
    "somoy tv": "somoynews.tv",
    "channel 24": "channel24bd.tv",
    "jamuna tv": "jamuna.tv",
    "jamuna": "jamuna.tv",
    "independent tv": "itvbd.com",
    "independent": "itvbd.com",
    "r tv": "rtvonline.com",
    "rtv": "rtvonline.com",
}


def _match_known_domain(outlet_name_or_text: str) -> str:
    """Returns the known real domain if outlet_name_or_text matches a KNOWN_OUTLET_DOMAINS
    key (case-insensitive substring match), else ''."""
    if not outlet_name_or_text:
        return ""
    lowered = outlet_name_or_text.lower()
    for key, domain in KNOWN_OUTLET_DOMAINS.items():
        if key in lowered:
            return domain
    return ""


# ==========================================
# TOOL 1: LLM Layout-Aware Photo Cropper (Method 2 Option A)
# ==========================================
@tool("Layout Aware News Photo Cropper Tool")
def crop_news_photo_llm(photocard_path: str) -> str:
    """Uses a vision LLM (local Qwen2.5-VL via Ollama in open-source mode, or Gemini in API mode) to crop news photo/cutout/poster grid area from photocard."""
    try:
        pil_img = Image.open(photocard_path)
        width, height = pil_img.size

        prompt = """
        Identify the primary news image, person cutout, poster grid, or visual content area in this photocard.
        Return ONLY a JSON object with relative coordinates [ymin, xmin, ymax, xmax] normalized from 0 to 1000:
        {"box_2d": [ymin, xmin, ymax, xmax]}
        """

        data = _call_vision_model(pil_img, prompt)
        box = data.get("box_2d", [100, 50, 700, 950])  # Fallback coordinates

        # De-normalize coordinates
        ymin = int((box[0] / 1000) * height)
        xmin = int((box[1] / 1000) * width)
        ymax = int((box[2] / 1000) * height)
        xmax = int((box[3] / 1000) * width)

        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(height, ymax), min(width, xmax)

        cropped = pil_img.crop((xmin, ymin, xmax, ymax))

        output_dir = "temp_crops"
        os.makedirs(output_dir, exist_ok=True)
        crop_save_path = os.path.join(output_dir, "cropped_news_photo.jpg")
        cropped.save(crop_save_path)

        return f"Photo cropped successfully and saved to: {crop_save_path}"

    except Exception as e:
        return f"Cropping failed: {str(e)}"

# ==========================================
# TOOL 2: Vision-LLM Bangla Headline Extraction
# ==========================================
# Reads the headline directly with the vision LLM instead of a separate OCR
# engine. Recent benchmarks (BanglaWild, DISCO 2026) show VLMs substantially
# outperform EasyOCR/Tesseract/PaddleOCR on stylized, non-scanned Bangla text
# like photocard headlines -- so this replaces OCR for headline reading.
# EasyOCR is still used below (Tool 4) for the narrow date-string task, where
# a classical OCR engine is fine.
@tool("Vision LLM Bangla Headline Extraction Tool")
def extract_headline_llm(photocard_path: str) -> str:
    """Uses a vision LLM (local Qwen2.5-VL via Ollama in open-source mode, or Gemini in API mode) to transcribe the exact Bangla headline text visible on the photocard, while excluding ad/sponsor content."""
    try:
        pil_img = Image.open(photocard_path)

        prompt = """
        This is a Bangla news photocard. It may contain THREE different kinds of text --
        distinguish them carefully:

        1. THE HEADLINE -- the main news title/claim. This is what you must extract.
        2. A SUB-PARAGRAPH / BODY TEXT -- sometimes a shorter explanatory sentence or
           paragraph appears below the headline, in smaller or lighter text. This is
           NOT the headline -- do not extract this.
        3. AD / SPONSOR CONTENT -- promotional text for products or services, almost
           always in a visually distinct colored banner/strip, typically at the very
           BOTTOM of the image. Common examples: mobile financial service ads
           (e.g. "বিকাশ", "নগদ" with phrases like "সেন্ড মানি", "ডিপিএস", "টাকা জমায়"),
           telecom balance/offer codes (e.g. "*৯#", "ইমার্জেন্সি ব্যালেন্স", operator
           logos), or product ads (e.g. appliance brands like "Walton", "Smart Fridge").
           This is NEVER the headline, even though it's real text on the image --
           EXCLUDE it completely.

        Extract ONLY item 1, the actual news headline, verbatim with no translation,
        correction, or paraphrasing. If you are unsure whether a piece of text is the
        headline or an ad, prefer the text that reads as a news claim/event (not a
        product name, app instruction, or promotional phrase) and that is NOT in a
        bottom colored banner.

        Return ONLY a JSON object:
        {"headline": "<exact Bangla headline text only, no ad/sponsor text>", "excluded_ad_text": "<any ad/sponsor text you identified and excluded, or empty string if none>"}
        """

        data = _call_vision_model(pil_img, prompt)
        headline = str(data.get("headline", "")).strip()
        excluded_ad_text = str(data.get("excluded_ad_text", "")).strip()

        if not headline:
            return "Headline Not Extracted"

        # IMPORTANT: return ONLY the clean headline text, with no annotations,
        # brackets, or commentary appended. An earlier version appended a
        # "[Excluded ad/sponsor text: ...]" note here, and downstream agents
        # mistakenly passed that annotation to the search tool instead of the
        # real Bangla headline -- producing searches for literal placeholder
        # text and false "no match anywhere" verdicts. The excluded ad text is
        # logged for debugging instead of being returned.
        if excluded_ad_text:
            print(f"[extract_headline_llm] Excluded ad/sponsor text (not returned): {excluded_ad_text[:100]!r}")
        return headline

    except Exception as e:
        return f"Headline extraction failed: {str(e)}"



# ==========================================
# TOOL 3: Outlet Branding Verifier (multi-variant logo + VLM recognition + website URL check)
# ==========================================
# canonical_logos folder layout -- ONE SUBFOLDER PER OUTLET, MULTIPLE LOGO
# VARIANTS ALLOWED per subfolder (e.g. text logo, icon-only logo):
#
#   canonical_logos/
#     ProthomAlo/
#       text_logo.png       <- e.g. Image 2 you shared: full wordmark
#       icon_only.png        <- e.g. Image 3 you shared: icon with no text
#     JamunaTV/
#       logo.png
#     DailyStarBangla/
#       logo.png
#     ...
#
# The tool checks the header region against EVERY variant under EVERY outlet
# folder and keeps the single best match across all of them -- so if an
# outlet's icon-only logo doesn't match but its text logo does (or vice
# versa), it still gets picked up.
CANONICAL_LOGO_FOLDER = "G:/News Photocard Research Thesis/Canonical Logo"


def _normalize_polarity(gray_img):
    """
    Normalizes a grayscale image so its background is consistently light
    (mean intensity >= 127), inverting it if the background is dark.

    Why this matters: SSIM compares raw pixel intensity patterns, so a
    white-text-on-black-background logo and a dark-text-on-white-background
    version of the SAME logo are close to inverted in pixel terms and will
    score as very DISSIMILAR under plain SSIM, even though they're the same
    logo. Normalizing both images to a consistent polarity before comparing
    fixes this. (Heuristic: background covers most of a logo image's area,
    so mean brightness is a reasonable proxy for background tone.)
    """
    if gray_img.mean() < 127:
        return 255 - gray_img
    return gray_img


def _auto_trim_to_content(gray_img, margin: int = 4, tolerance: int = 20):
    """
    Auto-crops away uniform background padding around a logo, estimated from
    the image's own corner pixels. This handles the common case where a
    circular/oval/arbitrary-shaped logo was cropped with a rectangular tool
    (e.g. a snipping tool) and necessarily includes extra background in the
    corners -- no manual pixel-perfect cropping needed on your end.
    `margin`: pixels of padding kept around the detected content, so edges
    aren't cut too aggressively. `tolerance`: how far a pixel's brightness
    must differ from the estimated background to count as "content".
    """
    h, w = gray_img.shape
    corners = [int(gray_img[0, 0]), int(gray_img[0, w - 1]), int(gray_img[h - 1, 0]), int(gray_img[h - 1, w - 1])]
    bg = int(np.mean(corners))
    diff = np.abs(gray_img.astype(int) - bg)
    mask = diff > tolerance
    coords = np.argwhere(mask)
    if coords.size == 0:
        return gray_img  # no clear content found (e.g. blank image) -- return unchanged
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(h, y1 + margin + 1)
    x1 = min(w, x1 + margin + 1)
    return gray_img[y0:y1, x0:x1]


def _load_grayscale_with_alpha_composited_white(image_path: str):
    """
    Loads an image as grayscale for SSIM comparison, correctly handling
    transparency: if the image has an alpha channel, it's composited onto a
    WHITE background first. cv2.imread(..., IMREAD_GRAYSCALE) silently drops
    alpha and can default transparent regions to black instead, which is
    wrong for logo assets with transparent backgrounds (common for official
    logo/press-kit downloads) and would distort the comparison unpredictably.
    """
    try:
        pil_img = Image.open(image_path)
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            pil_img = pil_img.convert("RGBA")
            white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            pil_img = Image.alpha_composite(white_bg, pil_img)
        return np.array(pil_img.convert("L"))
    except Exception:
        return None


LOGO_LOCATION_PROMPT = """
Locate the NEWS OUTLET's OWN logo or watermark anywhere in this photocard
image. Important: it is NOT always at the very top, and it is NOT
necessarily the largest or most prominent circular/badge graphic in the
image -- some photocards show a large logo belonging to the story's SUBJECT
(e.g. a company or organization being reported on), while the outlet's own
branding is a small text watermark tucked in a corner instead. Find the
outlet's OWN brand mark specifically, not the news photo, not the headline
text, and not any other organization's logo that may appear as part of the
story content.

Return ONLY a JSON object with relative coordinates [ymin, xmin, ymax, xmax]
normalized from 0 to 1000, tightly bounding just the outlet's own logo:
{"box_2d": [ymin, xmin, ymax, xmax]}
If you cannot find the outlet's own logo anywhere, return {"box_2d": null}.
"""


def _locate_logo_bbox(photocard_path: str, img_cv):
    """
    Uses the vision LLM to locate the outlet's own logo anywhere in the
    image (not assumed to be in a fixed top region -- see LOGO_LOCATION_PROMPT
    for why this matters). Returns (ymin, xmin, ymax, xmax) in pixel
    coordinates, or None if the model couldn't locate one (caller decides
    the fallback).
    """
    h, w, _ = img_cv.shape
    try:
        pil_img = Image.open(photocard_path)
        data = _call_vision_model(pil_img, LOGO_LOCATION_PROMPT)
        box = data.get("box_2d")
    except Exception:
        box = None

    if box and isinstance(box, list) and len(box) == 4:
        ymin = int((box[0] / 1000) * h)
        xmin = int((box[1] / 1000) * w)
        ymax = int((box[2] / 1000) * h)
        xmax = int((box[3] / 1000) * w)
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(h, ymax), min(w, xmax)
        if ymax > ymin and xmax > xmin:
            return (ymin, xmin, ymax, xmax)
    return None


def _locate_logo_region(photocard_path: str, img_cv):
    """
    Grayscale crop of the located logo region, for the SSIM path in
    _best_logo_match. Falls back to the old top-25%-of-width heuristic if
    _locate_logo_bbox couldn't find anything, so the tool never fails
    outright.
    """
    h, w, _ = img_cv.shape
    bbox = _locate_logo_bbox(photocard_path, img_cv)
    if bbox:
        ymin, xmin, ymax, xmax = bbox
        return cv2.cvtColor(img_cv[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2GRAY)
    top_header = img_cv[0:int(h * 0.25), 0:w]
    return cv2.cvtColor(top_header, cv2.COLOR_BGR2GRAY)


def _best_logo_match(photocard_path: str) -> tuple[str, str, float]:
    """
    Compares the vision-LLM-LOCATED logo region of the photocard (NOT a
    fixed top-25% assumption -- outlet logos can appear anywhere, and even
    the same outlet may use very different logo styles/positions across
    templates) against every logo variant image under every outlet subfolder
    in CANONICAL_LOGO_FOLDER, with both sides polarity-normalized and
    auto-trimmed before SSIM.

    NOTE on interpreting the score: even with correct localization, SSIM
    remains fragile to compression artifacts and small-scale rendering
    differences -- empirically (tested against real photocards), a correctly
    located crop of the true logo can still score well below a naive "0.85 =
    authentic" threshold. Treat the returned score as ONE weak-to-moderate
    supporting signal alongside the VLM recognition and website URL checks
    in verify_outlet_branding, not as a standalone pass/fail gate.

    Returns (outlet_folder_name, matched_filename, best_ssim_score).
    """
    img = cv2.imread(photocard_path)
    if img is None:
        return ("None", "None", 0.0)

    gray_header = _locate_logo_region(photocard_path, img)
    gray_header = _auto_trim_to_content(gray_header)
    gray_header = _normalize_polarity(gray_header)

    best_score = 0.0
    best_outlet = "None"
    best_file = "None"

    if not os.path.exists(CANONICAL_LOGO_FOLDER):
        return (best_outlet, best_file, best_score)

    for outlet_folder in os.listdir(CANONICAL_LOGO_FOLDER):
        outlet_path = os.path.join(CANONICAL_LOGO_FOLDER, outlet_folder)
        if not os.path.isdir(outlet_path):
            continue
        for logo_file in os.listdir(outlet_path):
            if not logo_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            logo_path = os.path.join(outlet_path, logo_file)
            canonical = _load_grayscale_with_alpha_composited_white(logo_path)
            if canonical is None:
                continue
            canonical = _auto_trim_to_content(canonical)
            canonical = _normalize_polarity(canonical)
            resized_canonical = cv2.resize(canonical, (gray_header.shape[1], gray_header.shape[0]))
            score, _ = ssim(gray_header, resized_canonical, full=True)
            if score > best_score:
                best_score = score
                best_outlet = outlet_folder
                best_file = logo_file

    return (best_outlet, best_file, best_score)


BRANDING_RECOGNITION_PROMPT = """
Look at this entire news photocard image and answer two things:

1. Based on the logo, wordmark, or brand styling visible (regardless of its
   background color, resolution, or any badge/checkmark next to it), which
   Bangladeshi news outlet's logo do you recognize this as, if any? Judge
   this the way a person familiar with Bangladeshi news brands would --
   by the text/wordmark and style, NOT by exact pixel appearance. If you
   don't recognize it with reasonable confidence, say so.
2. Is there any visible website address, URL, or domain text anywhere on
   the image (often in a footer or watermark, e.g. "prothomalo.com")?

Return ONLY a JSON object:
{"recognized_outlet_logo": "<outlet name you recognize, or empty string if unsure>", "website_text": "<exact website/URL text as shown, or empty string if none visible>"}
"""


@tool("Outlet Branding Verifier Tool")
def verify_outlet_branding(photocard_path: str) -> str:
    """
    Verifies the news outlet's branding through THREE independent signals
    and combines the evidence:
      1. Logo pixel match (SSIM): compares a vision-LLM-LOCATED logo region
         of the photocard (position is detected per-image, not assumed to be
         at a fixed spot -- logo placement and style vary a lot, even for
         the same outlet across different templates) against EVERY logo
         variant of EVERY outlet in the canonical_logos reference folder.
         IMPORTANT: tested empirically against real photocards, SSIM remains
         fragile even with correct localization (compression artifacts,
         small-scale rendering) -- treat this as a WEAK-TO-MODERATE signal,
         not a pass/fail authenticity gate. A low score here does NOT
         reliably mean the outlet is fake.
      2. Vision-LLM logo recognition: asks the model to identify the outlet
         by general appearance, the way a person would -- robust to
         background color, resolution, and badges/checkmarks, but won't
         catch a very subtle character-level forgery the way #1 can. This
         is the STRONGER of the two logo-based signals in practice.
      3. Website URL: reads any visible website address anywhere on the
         photocard and checks it against known real outlet domains.
    These three are complementary -- use all of them together, weighting
    #2 and #3 more heavily than #1's raw score, rather than trusting any
    single one.
    """
    outlet, logo_file, score = _best_logo_match(photocard_path)
    if score >= 0.70:
        logo_signal_strength = "strong"
    elif score >= 0.40:
        logo_signal_strength = "moderate"
    else:
        logo_signal_strength = "weak -- do NOT treat this alone as evidence of a fake; SSIM is known to score low even on correctly-matched real logos"

    try:
        pil_img = Image.open(photocard_path)
        data = _call_vision_model(pil_img, BRANDING_RECOGNITION_PROMPT)
        recognized_outlet = str(data.get("recognized_outlet_logo", "")).strip()
        website_text = str(data.get("website_text", "")).strip()
        vlm_error = ""
    except Exception as e:
        recognized_outlet = ""
        website_text = ""
        vlm_error = f" (VLM branding check failed: {e})"

    known_domain_match = (
        _match_known_domain(website_text)
        or _match_known_domain(recognized_outlet)
        or _match_known_domain(outlet)
    )

    return (
        f"[1] Logo pixel match (SSIM, vision-LLM-located region, polarity-normalized): "
        f"outlet='{outlet}' | variant file='{logo_file}' | score={score:.4f} | "
        f"signal strength={logo_signal_strength}\n"
        f"[2] Vision-LLM recognized outlet (appearance-based, ignores color/resolution -- "
        f"this is the more reliable of the two logo signals): "
        f"'{recognized_outlet or 'not recognized'}'{vlm_error}\n"
        f"[3] Website text detected on photocard: '{website_text or 'none'}'\n"
        f"Matches a known real outlet domain: "
        f"{known_domain_match if known_domain_match else 'no match / not in KNOWN_OUTLET_DOMAINS yet'}"
    )


def _resolve_outlet_folder(claimed_outlet_name: str):
    """
    Fuzzy-matches a claimed outlet name/text to an actual subfolder name in
    CANONICAL_LOGO_FOLDER (case-insensitive, whitespace-insensitive
    substring match in either direction, e.g. "Prothom Alo" <-> "ProthomAlo").
    Returns the real folder name, or None if nothing matches.
    """
    if not claimed_outlet_name or not os.path.exists(CANONICAL_LOGO_FOLDER):
        return None
    claimed_key = claimed_outlet_name.lower().replace(" ", "")
    for outlet_folder in os.listdir(CANONICAL_LOGO_FOLDER):
        if not os.path.isdir(os.path.join(CANONICAL_LOGO_FOLDER, outlet_folder)):
            continue
        folder_key = outlet_folder.lower().replace(" ", "")
        if folder_key in claimed_key or claimed_key in folder_key:
            return outlet_folder
    return None


LOGO_STYLE_COMPARISON_PROMPT_TEMPLATE = """
The FIRST image below is a logo cropped from a news photocard that claims
to be from "{outlet_name}".
The remaining {n_refs} image(s) are KNOWN AUTHENTIC reference logos for
"{outlet_name}", showing its real style/pattern/wording/colors (there may
be more than one valid reference style, e.g. a full text wordmark and a
separate icon-only mark -- a match to ANY reference counts as a match).

Compare the FIRST image against the reference image(s), the way a person
would visually compare two logos. Does it genuinely match the same brand --
same wording/spelling, same icon/symbol, same overall color scheme and
composition -- allowing for normal differences in image quality,
resolution, compression, exact scale/crop, and lighting? Look specifically
for signs of impersonation: altered or misspelled text, a wrong or extra
symbol, a different color scheme, or a fundamentally different design.

Return ONLY a JSON object:
{{"matches_reference_style": true or false, "closest_reference_index": <1-based index of the most similar reference image, or 0 if none are close>, "confidence": a number 0.0-1.0, "reasoning": "1-2 sentences on what matched or what looked off"}}
"""


@tool("Logo Style Comparison Tool (vs. claimed outlet's reference folder)")
def compare_logo_to_reference_style(photocard_path: str, claimed_outlet_name: str) -> str:
    """
    Directly asks the vision LLM to compare the photocard's logo against the
    KNOWN REFERENCE logo images collected for the CLAIMED outlet
    specifically -- a genuine visual/style judgment (wording, icon, color
    scheme, composition), not a raw pixel-distance computation. This is
    generally MORE reliable than the SSIM check in verify_outlet_branding,
    since it tolerates scale/compression/lighting differences the way a
    person would, rather than penalizing them as mismatches.

    claimed_outlet_name: the outlet name as identified so far (e.g. from
    verify_outlet_branding's recognized_outlet_logo, or your own reading of
    the photocard) -- this determines WHICH outlet's reference folder to
    compare against. Requires that folder to already contain reference logo
    image(s); if no matching folder is found, this returns a message saying
    so rather than a false result.

    NOTE: this embeds multiple images (the located logo crop + every
    reference image in that outlet's folder) in a single vision-LLM call,
    so it's slower/costlier per photocard than the other checks -- this is
    a worthwhile tradeoff given it's the stronger signal, but keep each
    outlet's reference folder to a reasonable number of files.
    """
    outlet_folder = _resolve_outlet_folder(claimed_outlet_name)
    if outlet_folder is None:
        return (
            f"No reference logo folder found matching claimed outlet "
            f"'{claimed_outlet_name}' -- cannot perform style comparison. "
            f"(Check canonical_logos/ folder names, or this outlet may "
            f"genuinely have no reference logos collected yet.)"
        )

    img_cv = cv2.imread(photocard_path)
    if img_cv is None:
        return "Error: could not load photocard image."

    h, w, _ = img_cv.shape
    bbox = _locate_logo_bbox(photocard_path, img_cv)
    if bbox:
        ymin, xmin, ymax, xmax = bbox
        crop_color = img_cv[ymin:ymax, xmin:xmax]
    else:
        crop_color = img_cv[0:int(h * 0.25), 0:w]  # fallback, same as the SSIM path
    crop_pil = Image.fromarray(cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB))

    outlet_path = os.path.join(CANONICAL_LOGO_FOLDER, outlet_folder)
    ref_files = [f for f in os.listdir(outlet_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not ref_files:
        return f"Reference folder '{outlet_folder}' exists but contains no logo image files."

    ref_images = []
    valid_ref_files = []
    for f in ref_files:
        try:
            ref_images.append(Image.open(os.path.join(outlet_path, f)).convert("RGB"))
            valid_ref_files.append(f)
        except Exception:
            continue

    prompt = LOGO_STYLE_COMPARISON_PROMPT_TEMPLATE.format(outlet_name=claimed_outlet_name, n_refs=len(ref_images))
    all_images = [crop_pil.convert("RGB")] + ref_images

    try:
        result = _call_vision_model_multi(all_images, prompt)
    except Exception as e:
        return f"Logo style comparison failed: {e}"

    matches = result.get("matches_reference_style")
    idx = result.get("closest_reference_index", 0)
    confidence = result.get("confidence", 0.0)
    reasoning = result.get("reasoning", "")
    closest_file = valid_ref_files[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(valid_ref_files) else "none"

    return (
        f"Style comparison against '{outlet_folder}' reference folder ({len(ref_images)} reference image(s)):\n"
        f"matches_reference_style={matches} | closest_reference='{closest_file}' | confidence={confidence}\n"
        f"reasoning: {reasoning}"
    )



# ==========================================
# TOOL 4: Date Extraction & ISO Normalizer
# (EasyOCR is fine here -- this is a narrow, structured extraction task,
#  unlike headline reading above)
# ==========================================
@tool("Bangla Date Extraction Tool")
def extract_bangla_date(photocard_path: str) -> str:
    """Extracts a Bangla date string from the photocard image and normalizes it to YYYY-MM-DD.
    Returns "Date Not Extracted" if no date is present -- this is expected in rare cases and
    is not a failure; proceed with headline+outlet verification alone in that case."""
    results = reader.readtext(photocard_path, detail=0)
    full_text = " ".join(results)

    BANGLA_DIGITS = {'০':'0', '১':'1', '২':'2', '৩':'3', '৪':'4', '৫':'5', '৬':'6', '৭':'7', '৮':'8', '৯':'9'}
    BANGLA_MONTHS = {
        'জানুয়ারি': '01', 'ফেব্রুয়ারি': '02', 'মার্চ': '03', 'এপ্রিল': '04',
        'মে': '05', 'জুন': '06', 'জুলাই': '07', 'আগস্ট': '08',
        'সেপ্টেম্বর': '09', 'অক্টোবর': '10', 'নভেম্বর': '11', 'ডিসেম্বর': '12'
    }

    month_alternation = '|'.join(re.escape(m) for m in BANGLA_MONTHS.keys())
    pattern = rf'([০-৯0-9]{{1,2}})\s*({month_alternation})\s*([০-৯0-9]{{4}})'
    match = re.search(pattern, full_text)

    if match:
        day_raw, month_raw, year_raw = match.groups()
        day = ''.join([BANGLA_DIGITS.get(c, c) for c in day_raw]).zfill(2)
        year = ''.join([BANGLA_DIGITS.get(c, c) for c in year_raw])
        month = BANGLA_MONTHS.get(month_raw, '01')
        return f"{year}-{month}-{day}"
    return "Date Not Extracted"

# ==========================================
# TOOL 5: Claimed-Outlet-First Search, with Misattribution-Aware Fallback
# ==========================================
WAYBACK_AVAILABLE_API = "https://archive.org/wayback/available"
SERPER_API_URL = "https://google.serper.dev/search"
SERPER_API_KEY_ENV_VAR = "SERPER_API_KEY"


def _wayback_closest_snapshot(domain: str, target_date: str = "") -> str | None:
    """
    Queries the Wayback Machine's Availability API for the closest archived
    snapshot of `domain` near `target_date` (YYYY-MM-DD). Returns the
    archived URL, or None if nothing is archived / the request fails.
    Used for the domain-level fallback (step 2) -- a DIFFERENT purpose from
    _wayback_backup_for_url below.
    """
    params = {"url": domain}
    if target_date:
        params["timestamp"] = target_date.replace("-", "")
    try:
        resp = requests.get(WAYBACK_AVAILABLE_API, params=params, timeout=10)
        resp.raise_for_status()
        snap = resp.json().get("archived_snapshots", {}).get("closest")
        if snap and snap.get("available"):
            return snap.get("url")
    except Exception:
        return None
    return None


def _wayback_backup_for_url(article_url: str) -> str | None:
    """
    Checks whether a SPECIFIC article URL (not just its domain) has an
    archived Wayback Machine backup. Used as COMPANION evidence once an
    article has already been found by search -- protects against link rot
    and gives a permanent copy of the page as it looked when found. This is
    NOT a discovery mechanism (Wayback has no full-text search), it just
    backs up a URL you already have.
    """
    if not article_url:
        return None
    try:
        resp = requests.get(WAYBACK_AVAILABLE_API, params={"url": article_url}, timeout=10)
        resp.raise_for_status()
        snap = resp.json().get("archived_snapshots", {}).get("closest")
        if snap and snap.get("available"):
            return snap.get("url")
    except Exception:
        return None
    return None


def _serper_search(query: str, num_results: int = 3) -> list[dict]:
    """
    Queries Serper.dev -- a Google SERP API (real Google search results,
    not an independent index like DuckDuckGo or Exa).

    Supports MULTIPLE API keys for rotation: set SERPER_API_KEYS (plural) to
    a comma-separated list, e.g. "key1,key2,key3,key4". If one key is
    exhausted/suspended (HTTP 401/403/429), automatically tries the next one
    rather than failing the whole run. Falls back to the single SERPER_API_KEY
    env var if SERPER_API_KEYS isn't set.

    Tracks total calls made (module-level counter, printed periodically) so
    you can watch real credit consumption during a run instead of finding out
    only when you check the dashboard afterward.

    Returns a list of {"title", "url", "snippet"} dicts. Raises if ALL keys
    fail -- callers catch this.
    """
    global _serper_call_count, _serper_key_index

    keys_raw = os.getenv("SERPER_API_KEYS", "") or os.getenv(SERPER_API_KEY_ENV_VAR, "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError(
            f"No Serper API key found. Set {SERPER_API_KEY_ENV_VAR} (single key) "
            f"or SERPER_API_KEYS (comma-separated, for multi-account rotation)."
        )

    last_error = None
    # Try keys starting from the current rotation index, wrapping around once.
    for attempt in range(len(keys)):
        idx = (_serper_key_index + attempt) % len(keys)
        api_key = keys[idx]
        try:
            resp = requests.post(
                SERPER_API_URL,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": num_results},
                timeout=15,
            )
            if resp.status_code in (401, 403, 429):
                print(f"[_serper_search] Key index {idx} returned {resp.status_code} "
                      f"(exhausted/suspended/rate-limited) -- rotating to next key.")
                last_error = f"HTTP {resp.status_code} on key index {idx}"
                continue
            resp.raise_for_status()

            _serper_key_index = idx  # stick with this key for subsequent calls
            _serper_call_count += 1
            if _serper_call_count % 50 == 0:
                print(f"[_serper_search] Usage so far: {_serper_call_count} calls "
                      f"(currently on key index {idx} of {len(keys)})")

            data = resp.json()
            return [
                {"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", "")}
                for item in data.get("organic", [])[:num_results]
            ]
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue

    raise RuntimeError(f"All {len(keys)} Serper API key(s) failed. Last error: {last_error}")


# Module-level state for key rotation and usage tracking (see _serper_search above)
_serper_call_count = 0
_serper_key_index = 0


@tool("Live + Archived + Misattribution-Aware News Search Tool")
def search_live_news_free(headline: str, outlet_domain: str = "", target_date: str = "") -> str:
    """
    Checks whether the CLAIMED outlet actually published this headline, and
    if not, distinguishes two different kinds of "not found" -- this
    distinction is what your RQ4 (source attribution impersonation) needs
    for error analysis:

      CASE A -- POSSIBLE MISATTRIBUTION: the story is found, but on a
        DIFFERENT real outlet, not the one claimed on the photocard. The
        event/quote may be genuine; the claimed outlet just didn't publish
        this graphic. This is source impersonation, not fabrication.
      CASE B -- NO MATCH ANYWHERE: the story isn't found on the claimed
        outlet or anywhere else on the open web. This suggests fabricated
        content rather than misattribution.

    In BOTH cases the verdict for the claimed outlet should still be Fake --
    this tool only tags WHICH kind of Fake it is, for your rationale/error
    analysis. It does not change the binary prediction.

    Search backend: Serper.dev, a real Google Search Results (SERP) API --
    same search engine for BOTH the Qwen and Gemini agentic arms, so your
    baseline-vs-agentic comparison isolates the model, not the search index.
    Requires the SERPER_API_KEY environment variable.

    Steps, in order:
      1. Site-restricted Google search on the claimed outlet's own domain
         (if given) -- does the claimed outlet itself show any record of
         this? If a URL is found, also checks whether that EXACT article URL
         has a Wayback Machine backup (companion evidence, protects against
         the live link going dead later -- NOT a discovery mechanism).
      2. Wayback Machine closest-snapshot check on the claimed outlet's
         DOMAIN (if given) -- catches older, unindexed content the claimed
         outlet may have published, when step 1 found nothing at all.
      3. ONLY if steps 1 and 2 both found nothing (or outlet_domain is
         unknown): one broad, domain-UNRESTRICTED Google search, to check
         whether this story exists anywhere at all, tagging the result as
         Case A or Case B. Also checks the found URL's Wayback backup, same
         as step 1.

    Args:
        headline: the exact Bangla headline text extracted from the photocard.
        outlet_domain: the claimed outlet's domain if known/inferable from
            its name (e.g. "prothomalo.com"). Pass "" if unknown -- steps 1
            and 2 will be skipped, and step 3 runs directly.
        target_date: the photocard's claimed date in YYYY-MM-DD format, if
            extracted. Pass "" if unavailable.

    IMPORTANT re: step 2 -- the Wayback Machine does not support full-text
    search across archived pages. A step-2 result only confirms the claimed
    outlet's site was live/archived around that time, not that this exact
    headline was published there. Treat it as weaker, supporting evidence.
    """
    # --- INPUT VALIDATION -------------------------------------------------
    # Guard against the agent passing placeholder/annotation text instead of
    # the real Bangla headline (this actually happened: searches were run for
    # the literal string "[Bengali Headline Excluded]", producing false
    # "no match anywhere" verdicts on genuinely real photocards).
    headline = (headline or "").strip()
    has_bangla = any('\u0980' <= ch <= '\u09FF' for ch in headline)
    looks_like_placeholder = (
        not headline
        or (headline.startswith("[") and headline.endswith("]"))
        or "headline excluded" in headline.lower()
        or "not extracted" in headline.lower()
        or "excluded ad" in headline.lower()
    )
    if looks_like_placeholder or not has_bangla:
        return (
            f"SEARCH NOT PERFORMED -- INVALID QUERY. The headline passed to this tool was "
            f"{headline[:120]!r}, which does not look like a real Bangla headline "
            f"(contains_bangla_characters={has_bangla}). Re-read the photocard and pass the "
            f"EXACT Bangla headline text verbatim -- do NOT pass placeholders, annotations, "
            f"English descriptions, or bracketed notes. This is NOT evidence that the story "
            f"doesn't exist, so do NOT treat it as a 'no match' result."
        )

    # Warn (but proceed) if the domain looks implausible for a Bangladeshi outlet.
    if outlet_domain:
        outlet_domain = outlet_domain.strip().replace("https://", "").replace("http://", "").rstrip("/")
        if outlet_domain not in KNOWN_OUTLET_DOMAINS.values():
            print(
                f"[search_live_news_free] WARNING: outlet_domain {outlet_domain!r} is not in "
                f"KNOWN_OUTLET_DOMAINS. If this is a guess, it may be wrong -- known real "
                f"domains are: {sorted(set(KNOWN_OUTLET_DOMAINS.values()))}"
            )

    print(f"[search_live_news_free] Searching headline={headline[:80]!r} domain={outlet_domain!r} date={target_date!r}")

    site_error = None

    # Step 1: site-restricted Google search (via Serper) on the claimed outlet's own domain
    if outlet_domain:
        try:
            query = f"site:{outlet_domain} {headline}"
            results = _serper_search(query, num_results=3)
            if results:
                top_url = results[0]["url"]
                backup_url = _wayback_backup_for_url(top_url)
                formatted = "\n---\n".join(
                    f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for r in results
                )
                backup_note = f"\nWayback backup of top result: {backup_url}" if backup_url else "\nWayback backup of top result: not archived"
                return (
                    f"[CONFIRMED ON CLAIMED OUTLET -- site-restricted match on {outlet_domain}]\n"
                    f"{formatted}{backup_note}"
                )
        except Exception as e:
            site_error = f"Site-restricted search failed: {str(e)}"
    else:
        site_error = "No outlet_domain provided -- site-restricted search skipped."

    # Step 2: Wayback Machine -- closest archived snapshot of the claimed outlet's DOMAIN
    # (only reached if step 1 found nothing at all)
    if outlet_domain:
        snapshot_url = _wayback_closest_snapshot(outlet_domain, target_date)
        if snapshot_url:
            return (
                f"[WEAK SUPPORTING EVIDENCE -- Wayback snapshot of {outlet_domain} near "
                f"{target_date or 'an unknown date'}]\n{snapshot_url}\n"
                f"Confirms the claimed outlet's site existed around this time, but does NOT "
                f"confirm this exact headline was published there. Treat as weak evidence only, "
                f"and still consider running step 3 reasoning if this is the only signal found."
            )

    # Step 3: broad, domain-UNRESTRICTED Google search -- last resort, only reached if
    # the claimed outlet showed nothing in steps 1-2. Purpose: distinguish
    # "story exists elsewhere" (misattribution) from "story exists nowhere"
    # (likely fabrication).
    try:
        broad_results = _serper_search(headline, num_results=5)
        broad_error = None
    except Exception as e:
        broad_results = []
        broad_error = f"Broad search failed: {str(e)}"

    if broad_results:
        matched_on_claimed_outlet = bool(
            outlet_domain and any(outlet_domain.lower() in r.get("url", "").lower() for r in broad_results)
        )
        top_url = broad_results[0]["url"]
        backup_url = _wayback_backup_for_url(top_url)
        backup_note = f"\nWayback backup of top result: {backup_url}" if backup_url else "\nWayback backup of top result: not archived"
        formatted = "\n---\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for r in broad_results[:3]
        )
        if matched_on_claimed_outlet:
            return f"[CONFIRMED ON CLAIMED OUTLET -- found via broad search]\n{formatted}{backup_note}"
        return (
            f"[CASE A: POSSIBLE MISATTRIBUTION -- story found, but NOT on the claimed outlet"
            f"{f' ({outlet_domain})' if outlet_domain else ''}]\n"
            f"The headline matches real coverage on a DIFFERENT outlet:\n{formatted}{backup_note}\n"
            f"This suggests the event/quote may be genuine, but the claimed outlet did not "
            f"publish this graphic -- likely source impersonation, not fabrication. Still "
            f"predict Fake for the claimed outlet, but reflect this distinction in the rationale."
        )

    return (
        f"[CASE B: NO MATCH ANYWHERE -- not found on the claimed outlet or anywhere else on the "
        f"open web]\nNo evidence found via site-restricted search, Wayback archive, or broad web "
        f"search. This suggests possible fabrication rather than misattribution. "
        f"({site_error or ''} {broad_error or ''})".strip()
    )