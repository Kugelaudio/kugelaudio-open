"""
Language configuration for KugelAudio Open.

Structured metadata for the 24 supported European languages with quality
tiers based on YODAS2 training data coverage.

Ref: https://github.com/Kugelaudio/kugelaudio-open/issues/10
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-1
    name: str          # English name
    native_name: str   # Endonym
    flag: str
    tier: str          # "high" | "medium" | "limited"

# Quality tiers reflect YODAS2 dataset representation (~200k hours).
LANGUAGES: Dict[str, Language] = {
    "en": Language("en", "English",    "English",      "🇺🇸", "high"),
    "de": Language("de", "German",     "Deutsch",      "🇩🇪", "high"),
    "fr": Language("fr", "French",     "Français",     "🇫🇷", "high"),
    "es": Language("es", "Spanish",    "Español",      "🇪🇸", "high"),
    "it": Language("it", "Italian",    "Italiano",     "🇮🇹", "medium"),
    "pt": Language("pt", "Portuguese", "Português",    "🇵🇹", "medium"),
    "nl": Language("nl", "Dutch",      "Nederlands",   "🇳🇱", "medium"),
    "pl": Language("pl", "Polish",     "Polski",       "🇵🇱", "medium"),
    "ru": Language("ru", "Russian",    "Русский",      "🇷🇺", "medium"),
    "uk": Language("uk", "Ukrainian",  "Українська",   "🇺🇦", "medium"),
    "cs": Language("cs", "Czech",      "Čeština",      "🇨🇿", "medium"),
    "ro": Language("ro", "Romanian",   "Română",       "🇷🇴", "limited"),
    "hu": Language("hu", "Hungarian",  "Magyar",       "🇭🇺", "limited"),
    "sv": Language("sv", "Swedish",    "Svenska",      "🇸🇪", "limited"),
    "da": Language("da", "Danish",     "Dansk",        "🇩🇰", "limited"),
    "fi": Language("fi", "Finnish",    "Suomi",        "🇫🇮", "limited"),
    "no": Language("no", "Norwegian",  "Norsk",        "🇳🇴", "limited"),
    "el": Language("el", "Greek",      "Ελληνικά",     "🇬🇷", "limited"),
    "bg": Language("bg", "Bulgarian",  "Български",    "🇧🇬", "limited"),
    "sk": Language("sk", "Slovak",     "Slovenčina",   "🇸🇰", "limited"),
    "hr": Language("hr", "Croatian",   "Hrvatski",     "🇭🇷", "limited"),
    "sr": Language("sr", "Serbian",    "Српски",       "🇷🇸", "limited"),
    "tr": Language("tr", "Turkish",    "Türkçe",       "🇹🇷", "limited"),
}

# Display order: high-quality first, then alphabetical within tiers
DISPLAY_ORDER: List[str] = [
    "en", "de", "fr", "es",
    "cs", "it", "nl", "pl", "pt", "ru", "uk",
    "bg", "da", "el", "fi", "hr", "hu", "no", "ro", "sk", "sr", "sv", "tr",
]

DEFAULT_LANG = "en"


def get(code: str) -> Optional[Language]:
    """Look up language by ISO 639-1 code. Case-insensitive."""
    return LANGUAGES.get(code.lower().strip())


def codes() -> List[str]:
    """All supported language codes in display order."""
    return list(DISPLAY_ORDER)


def validate(code: str) -> str:
    """Validate and normalize a language code. Raises ValueError if unsupported."""
    c = code.lower().strip()
    if c not in LANGUAGES:
        raise ValueError(
            f"Unsupported language: '{code}'. "
            f"Supported: {', '.join(codes())}"
        )
    return c


def quality_warning(code: str) -> Optional[str]:
    """Return a warning string for limited-tier languages, None otherwise."""
    lang = get(code)
    if lang and lang.tier == "limited":
        return (
            f"⚠️ {lang.name} has limited training data. "
            f"Best quality: en, de, fr, es."
        )
    return None


# ── Gradio helpers ────────────────────────────────────────────────────────────

def gradio_choices() -> List[str]:
    """Formatted strings for Gradio dropdown: '🇩🇪 German (de)'."""
    out = []
    for c in DISPLAY_ORDER:
        lang = LANGUAGES[c]
        warn = " ⚠️" if lang.tier == "limited" else ""
        out.append(f"{lang.flag} {lang.name} ({c}){warn}")
    return out


def parse_gradio_choice(choice: str) -> str:
    """Extract language code from Gradio dropdown value."""
    try:
        return choice.split("(")[-1].split(")")[0].strip()
    except (IndexError, AttributeError):
        return DEFAULT_LANG
