"""Mention identification for immigration speeches.

For each sentence we want to find references to immigrants (direct mentions
like "immigrants", group terms like "Mexicans", or nationality+role phrases
like "Mexican laborers") and produce a single-token [MASK] version suitable
for BERT MLM scoring.

For nationality+role matches we mask only the head noun (Card et al. 2022),
since BERT's [MASK] is one token and replacing a two-token span breaks the
surrounding grammar.
"""

import re
from dataclasses import dataclass


DIRECT_MENTION_TERMS = [
    "immigrant", "immigrants",
    "alien", "aliens",
    "migrant", "migrants",
    "refugee", "refugees",
    "foreigner", "foreigners",
    "newcomer", "newcomers",
    "stranger", "noncitizen",
    "displaced person", "displaced persons",
    "asylum seeker", "asylum seekers",
]

NATIONALITY_TERMS = [
    "mexican", "mexicans", "chinese", "italian", "italians", "irish",
    "german", "germans", "japanese", "cuban", "cubans", "haitian", "haitians",
    "polish", "russian", "russians", "soviet",
    "european", "europeans", "asian", "asians",
    "latin american", "latin americans",
    "central american", "central americans",
    "south american", "south americans",
    "afghan", "albanian", "algerian", "andorran", "angolan", "anguillan",
    "antiguan", "argentine", "armenian", "australian", "austrian", "azerbaijani",
    "bahamian", "bahraini", "bangladeshi", "barbadian", "belarusian", "belgian",
    "belizean", "beninese", "bermudian", "bhutanese", "bolivian", "bosnian",
    "botswanan", "brazilian", "british", "bruneian", "bulgarian", "burkinan",
    "burmese", "burundian",
    "cambodian", "cameroonian", "canadian", "cape verdean", "cayman islander",
    "central african", "chadian", "chilean", "colombian", "comoran", "congolese",
    "cook islander", "costa rican", "croatian", "cypriot", "czech",
    "danish", "djiboutian", "dominican", "dutch",
    "east timorese", "ecuadorean", "egyptian", "emirati", "english",
    "equatorial guinean", "eritrean", "estonian", "ethiopian",
    "faroese", "fijian", "filipino", "finnish", "french",
    "gabonese", "gambian", "georgian", "ghanaian", "gibraltarian", "greek",
    "greenlandic", "grenadian", "guamanian", "guatemalan", "guinean", "guyanese",
    "honduran", "hong konger", "hungarian",
    "icelandic", "indian", "indonesian", "iranian", "iraqi", "israeli", "ivorian",
    "jamaican", "jordanian",
    "kazakh", "kenyan", "kittitian", "kiribati", "kosovan", "kuwaiti", "kyrgyz",
    "lao", "latvian", "lebanese", "liberian", "libyan", "lithuanian", "luxembourger",
    "macanese", "macedonian", "malagasy", "malawian", "malaysian", "maldivian",
    "malian", "maltese", "marshallese", "martiniquais", "mauritanian", "mauritian",
    "micronesian", "moldovan", "monegasque", "mongolian", "montenegrin",
    "montserratian", "moroccan", "mosotho", "mozambican",
    "namibian", "nauruan", "nepalese", "new zealander", "nicaraguan", "nigerian",
    "nigerien", "niuean", "north korean", "northern irish", "norwegian",
    "omani",
    "pakistani", "palauan", "palestinian", "panamanian", "papua new guinean",
    "paraguayan", "peruvian", "pitcairn islander", "portuguese", "puerto rican",
    "qatari",
    "romanian", "rwandan",
    "salvadorean", "sammarinese", "samoan", "sao tomean", "saudi arabian",
    "scottish", "senegalese", "serbian", "seychellois", "sierra leonean",
    "singaporean", "slovak", "slovenian", "solomon islander", "somali",
    "south african", "south korean", "south sudanese", "spanish", "sri lankan",
    "st helenian", "st lucian", "sudanese", "surinamese", "swazi", "swedish",
    "swiss", "syrian",
    "taiwanese", "tajik", "tanzanian", "thai", "togolese", "tongan", "trinidadian",
    "tristanian", "tunisian", "turkish", "turkmen", "turks and caicos islander",
    "tuvaluan",
    "ugandan", "ukrainian", "uruguayan", "uzbek",
    "vatican", "vanuatuan", "venezuelan", "vietnamese", "vincentian",
    "wallisian", "welsh",
    "yemeni",
    "zambian", "zimbabwean",
]

PERSON_ROLE_TERMS = [
    "national", "nationals",
    "worker", "workers",
    "laborer", "laborers",
    "farmworker", "farmworkers",
    "farmer", "farmers",
    "person", "persons",
    "people",
    "family", "families",
    "child", "children",
    "student", "students",
    "man", "men",
    "woman", "women",
]


@dataclass
class Mention:
    masked_sentence: str
    mention_text: str
    mention_type: str
    sentence: str
    char_start: int
    char_end: int


def _alt(terms):
    return "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))


def build_mention_regex() -> re.Pattern:
    pattern = (
        rf"\b(?P<nat_role_nat>{_alt(NATIONALITY_TERMS)})\s+(?P<nat_role_role>{_alt(PERSON_ROLE_TERMS)})\b"
        r"|"
        rf"\b(?P<direct>{_alt(DIRECT_MENTION_TERMS)})\b"
        r"|"
        rf"\b(?P<nat_only>{_alt(NATIONALITY_TERMS)})\b"
    )
    return re.compile(pattern, flags=re.IGNORECASE)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def find_masked_sentences(text: str, mention_pattern: re.Pattern, mask_token: str = "[MASK]") -> list[Mention]:
    """For each mention in each sentence, return a Mention with the [MASK]ed
    sentence. For nationality+role matches we mask only the role (head noun),
    keeping the nationality as context (Card et al. 2022)."""
    out: list[Mention] = []
    for sentence in split_sentences(text):
        for match in mention_pattern.finditer(sentence):
            groups = match.groupdict()
            if groups["nat_role_role"] is not None:
                role_start = match.start("nat_role_role")
                role_end = match.end("nat_role_role")
                masked = sentence[:role_start] + mask_token + sentence[role_end:]
                mtype = "nat_role"
                mtext = match.group(0)
            elif groups["direct"] is not None:
                masked = sentence[:match.start()] + mask_token + sentence[match.end():]
                mtype = "direct"
                mtext = match.group(0)
            else:
                masked = sentence[:match.start()] + mask_token + sentence[match.end():]
                mtype = "nationality"
                mtext = match.group(0)
            out.append(Mention(
                masked_sentence=masked,
                mention_text=mtext,
                mention_type=mtype,
                sentence=sentence,
                char_start=match.start(),
                char_end=match.end(),
            ))
    return out
