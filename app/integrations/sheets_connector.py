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

    except HttpError as error:
        logging.error(f"Error creating sheet '{sheet_name}': {error}")
        raise


def _add_headers_to_sheet(service, sheet_name: str):
    DEFAULT_HEADERS = {
        "trend_scores": [
            "timestamp", "query", "score"
        ],
        "generated_content": [
            "timestamp",
            "id",
            "prompt",
            "original_content",
            "optimized_content",
            "variant",
            "score",
            "trend_score",
            "sentiment_score"
        ],
        "sentiment_results": [
            "timestamp", "text", "sentiment_label",
            "sentiment_score", "polarity", "emotions",
            "language", "trend_score"
        ],
        "raw_feedback": [
            "timestamp", "id", "source", "text",
            "sentiment_label", "sentiment_score",
            "polarity", "emotions",
            "trend_score_model", "trend_score_engine"
        ],
        "aggregates": [
            "timestamp", "total", "avg_score",
            "pos_count", "neg_count", "neu_count",
            "pct_positive", "pct_negative",
            "avg_toxicity", "dominant_emotion"
        ],
        "campaign_logs": [
            "timestamp", "event", "info"
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
