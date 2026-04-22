"""
Mock Lead Capture Tool.

This simulates persisting a qualified lead to a CRM or backend database.
It is called ONLY after all three required fields (name, email, platform)
have been collected from the user.
"""

from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates capturing a qualified lead.

    Args:
        name:     Full name of the lead.
        email:    Email address.
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        A confirmation string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simulate CRM write
    print("\n" + "=" * 60)
    print("🎯  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Name      : {name}")
    print(f"  Email     : {email}")
    print(f"  Platform  : {platform}")
    print(f"  Timestamp : {timestamp}")
    print("=" * 60 + "\n")

    return f"Lead captured successfully: {name}, {email}, {platform}"
