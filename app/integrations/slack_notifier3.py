# slack_notifier.py (WEBHOOK-ONLY VERSION)
"""
SlackNotifier — Webhook-Only Version

Requirements:
------------
- SLACK_WEBHOOK_URL must be present in .env
"""

import os
import json
import requests
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)


class SlackNotifier:
    """
    Sends messages to Slack using Webhook URL.
    If no webhook is configured, it safely does nothing.
    """

    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if self.webhook_url:
            self.mode = "webhook"
        else:
            self.mode = "disabled"

        logger.info(f"SlackNotifier initialized (mode = {self.mode})")

    # ------------------------------------------------------
    # Public method to send Slack message
    # ------------------------------------------------------
    def send_message(self, message: str) -> bool:
        """
        Sends a plain text message to Slack using the webhook.
        Returns True if successful, False otherwise.
        """

        if self.mode != "webhook":
            logger.warning("SlackNotifier disabled — No webhook URL set.")
            return False

        try:
            payload = {"text": message}

            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info("Slack message sent successfully.")
                return True
            else:
                logger.error(
                    f"Slack webhook error {response.status_code}: {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Slack webhook send failed: {e}")
            return False


# ------------------------------------------------------
# Manual Test (optional)
# ------------------------------------------------------
if __name__ == "__main__":
    notifier = SlackNotifier()
    notifier.send_message("Test message from Webhook-Only SlackNotifier 🚀")
