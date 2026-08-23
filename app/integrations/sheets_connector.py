import os
import logging
from typing import List, Any, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

# -----------------------------------------------------------
# Google Sheets Connector Module
# -----------------------------------------------------------
# Purpose:
#   - Unified wrapper for all Google Sheets interactions
#   - Used by metrics tracker, AB coach, run.py, etc.
# -----------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
CREDENTIALS_FILE = "credentials/service_account.json"


# ---------------------------------------------
# Load Credentials
# ---------------------------------------------
from google.oauth2.service_account import Credentials as ServiceAccountCredentials

def load_credentials():
    """Loads Google Service Account credentials (recommended for backend apps)."""
    if not os.path.exists(CREDENTIALS_FILE):
        raise FileNotFoundError(f"{CREDENTIALS_FILE} not found.")

    creds = ServiceAccountCredentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=SCOPES
    )
    return creds


# ---------------------------------------------
# Google Sheets Service
# ---------------------------------------------
def get_service():
    """Returns Google Sheets API service instance."""
    creds = load_credentials()
    try:
        service = build("sheets", "v4", credentials=creds)
        return service
    except Exception as e:
        logging.error(f"Failed to initialize Google Sheets service: {e}")
        raise


# ---------------------------------------------
# Ensure Sheet Exists
# ---------------------------------------------
def create_sheet_if_not_exists(sheet_name: str) -> None:
    """Creates a new worksheet tab if not exists."""
    try:
        service = get_service()
        spreadsheet = service.spreadsheets().get(spreadsheetId=SHEET_ID).execute()

        sheet_titles = [s["properties"]["title"] for s in spreadsheet.get("sheets", [])]

        if sheet_name in sheet_titles:
            # Check if header row exists
            result = service.spreadsheets().values().get(
                spreadsheetId=SHEET_ID,
                range=f"{sheet_name}!A1:Z1"
            ).execute()

            # If header row is missing → add header row
            if "values" not in result or not result["values"]:
                _add_headers_to_sheet(service, sheet_name)

            return

        request_body = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {
                            "title": sheet_name
                        }
                    }
                }
            ]
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=SHEET_ID,
            body=request_body
        ).execute()
        logging.info(f"Sheet '{sheet_name}' created successfully.")

        # A freshly-created sheet has no header row yet either -- the old
        # code only called _add_headers_to_sheet() on the "sheet already
        # existed but had no header" branch above, never on first creation.
        # That left brand-new, append-only sheets (ab_schedule, ab_posts,
        # etc.) header-less indefinitely.
        _add_headers_to_sheet(service, sheet_name)

    except HttpError as error:
        logging.error(f"Error creating sheet '{sheet_name}': {error}")
        raise


def _add_headers_to_sheet(service, sheet_name: str):
    DEFAULT_HEADERS = {
        "trend_scores": [
            # Matches trend_based_optimizer.py's _log_to_sheets():
            # append_row(sheet_name, [ts, preview, trend_score, keywords_str])
            "timestamp", "content_preview", "trend_score", "trending_keywords"
        ],
        "generated_content": [
            # Matches content_generator.py's generate_final_variations():
            # append_row("generated_content", [ts, platform, topic[:40]+"...",
            #                                   optimized_text[:80]+"...", trend_score])
            "timestamp", "platform", "topic_preview", "content_preview", "trend_score"
        ],
        "sentiment_results": [
            # FIXED: was an 8-col header (incl. timestamp, emotions, language)
            # that didn't match what's actually written. sentiment_analyzer.py's
            # analyze_sentiment() only ever sends:
            #   _safe_append_row("sentiment_results", [preview, label, norm_score, polarity, trend_score])
            # -- 5 values, no timestamp, no emotions, no language.
            "text_preview", "sentiment_label", "sentiment_score", "polarity", "trend_score"
        ],
        "raw_feedback": [
            # Matches tracker3.py's push_raw_feedback() row exactly (9 values,
            # single trend_score column -- the old duplicate
            # trend_score_model/trend_score_engine header was the original bug).
            "timestamp", "id", "source", "text",
            "sentiment_label", "sentiment_score",
            "polarity", "emotions", "trend_score"
        ],
        "aggregates": [
            # Confirmed against tracker3.py's push_aggregates() -- 10 values,
            # order matches exactly. No change needed.
            "timestamp", "total", "avg_score",
            "pos_count", "neg_count", "neu_count",
            "pct_positive", "pct_negative",
            "avg_toxicity", "dominant_emotion"
        ],
        "campaign_logs": [
            # Confirmed against tracker3.py's log_campaign_event() -- 3 values,
            # order matches exactly. No change needed.
            "timestamp", "event", "info"
        ],
        "ab_schedule": [
            # RESOLVED (Bug 2): social_poster.schedule_ab_test() is now the
            # sole writer -- ab_coach._persist_schedule() was deleted, since
            # it duplicated the same row a moment later in a different
            # column order. This header matches social_poster's write:
            #   [ts, ab_id, campaign_id, jobA, run_date_A, jobB, run_date_B, eval_time]
            "timestamp", "ab_id", "campaign_id", "job_a", "run_date_a", "job_b", "run_date_b", "eval_time"
        ],
        "ab_posts": [
            # Confirmed identical 5-col shape from ab_coach._persist_ab_posts()
            # and social_poster._ab_post_A/_B().
            "timestamp", "ab_id", "campaign_id", "variant", "post_id"
        ],
        "ab_test_results": [
            # FIXED: column names must match auto_retrainer.py's
            # REQUIRED_AB_COLUMNS = {"postA", "scoreA", "postB", "scoreB", "winner"}
            # EXACTLY (case-sensitive) -- load_training_data() does
            # REQUIRED_AB_COLUMNS.issubset(df.columns) against this header.
            # The previous snake_case version (post_a, score_a...) would
            # never match, so AutoRetrainer would always log "missing
            # required columns" and silently return an empty DataFrame --
            # retraining would never run, with no crash to reveal it.
            "timestamp", "ab_id", "postA", "scoreA", "postB", "scoreB", "winner"
        ],
        "campaign_ab_summary": [
            # RESOLVED (Bug 3): tracker.py's push_ab_test_results() used to
            # write this 8-col, campaign/variant-based row to "ab_test_results",
            # colliding with the ab_id-based schema ab_coach.py/social_poster.py
            # own above. Moved to its own sheet -- see tracker.py.
            "timestamp", "campaign_id", "variant", "impressions", "clicks", "conversions", "ctr", "conv_rate"
        ],
        "posted_content": [
            # Confirmed from social_poster._execute_and_persist_post():
            # row = [ts, campaign_id, variant, post_id, "twitter", text]
            "timestamp", "campaign_id", "variant", "post_id", "platform", "text"
        ],
        "model_versions": [
            # FIXED: was a 5-col guess. auto_retrainer.py's save_model() only
            # ever writes:
            #   append_row(self.config.sheet_model_versions, [timestamp, len(df), versioned_path])
            # -- 3 values.
            "timestamp", "num_samples", "model_path"
        ],
        "comment_sentiment": [
            # FIXED: was a 4-col guess with a leading timestamp.
            # sentiment_analyzer.py's analyze_post_comments() only ever writes:
            #   _safe_append_row("comment_sentiment", [post_id, avg_sent, avg_pol, avg_toxic, json.dumps(labels)])
            # -- 5 values, no timestamp.
            "post_id", "avg_sentiment", "avg_polarity", "avg_toxicity", "labels"
        ],
        "campaigns": [
            # NEW: metrics_hub.py's record_campaign_metrics() writes to a
            # sheet called "campaigns" (distinct from "campaign_logs") that
            # wasn't in DEFAULT_HEADERS at all -- 13 values:
            #   append_row("campaigns", [ts, campaign_id, variant, post_id, platform,
            #     impressions, clicks, conversions, ctr, conv_rate,
            #     sentiment_score, trend_score, avg_comment_sentiment])
            "timestamp", "campaign_id", "variant", "post_id", "platform",
            "impressions", "clicks", "conversions", "ctr", "conv_rate",
            "sentiment_score", "trend_score", "avg_comment_sentiment"
        ],
    }

    headers = DEFAULT_HEADERS.get(sheet_name, ["timestamp", "data"])

    service.spreadsheets().values().update(
        spreadsheetId=SHEET_ID,
        range=f"{sheet_name}!A1",
        valueInputOption="RAW",
        body={"values": [headers]},
    ).execute()

    logging.info(f"Headers added to sheet: {sheet_name}")


# ---------------------------------------------
# Append Row
# ---------------------------------------------
def append_row(sheet_name: str, row_data: List[Any]) -> None:
    """Appends a row to the specified sheet."""
    try:
        create_sheet_if_not_exists(sheet_name)
        service = get_service()

        service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=f"{sheet_name}!A1",
            valueInputOption="RAW",
            body={"values": [row_data]},
        ).execute()

        logging.info(f"Row appended to sheet '{sheet_name}'.")

    except HttpError as error:
        logging.error(f"Error appending row to sheet '{sheet_name}': {error}")
        raise


# ---------------------------------------------
# Read Rows
# ---------------------------------------------
def read_rows(sheet_name: str) -> List[List[Any]]:
    try:
        create_sheet_if_not_exists(sheet_name)
        service = get_service()

        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID, range=f"{sheet_name}!A1:Z"
        ).execute()

        return result.get("values", [])

    except HttpError as error:
        logging.error(f"Error reading rows from sheet '{sheet_name}': {error}")
        raise


# ---------------------------------------------
# Update Row
# ---------------------------------------------
def update_row(sheet_name: str, row_index: int, row_data: List[Any]) -> None:
    """Replaces entire row at given index (1-based)."""
    try:
        create_sheet_if_not_exists(sheet_name)
        service = get_service()

        service.spreadsheets().values().update(
            spreadsheetId=SHEET_ID,
            range=f"{sheet_name}!A{row_index}",
            valueInputOption="RAW",
            body={"values": [row_data]},
        ).execute()

        logging.info(f"Row {row_index} updated in sheet '{sheet_name}'.")

    except HttpError as error:
        logging.error(f"Error updating row in sheet '{sheet_name}': {error}")
        raise


# ---------------------------------------------
# Find Row by Keyword
# ---------------------------------------------
def find_row(sheet_name: str, keyword: str) -> Optional[int]:
    """Returns row index containing the keyword, else None."""
    try:
        rows = read_rows(sheet_name)
        for index, row in enumerate(rows, start=1):
            if keyword in str(row):
                return index
        return None

    except Exception as e:
        logging.error(f"Error searching row in sheet '{sheet_name}': {e}")
        return None