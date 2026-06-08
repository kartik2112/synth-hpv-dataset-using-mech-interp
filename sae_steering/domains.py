"""
HPV subtopic ("domain") definitions.

Each domain has a handful of short PROBE sentences. We push these through the
model and read the SAE feature activations to discover the features that light
up for the domain — those features' decoder directions become the "topic vector"
we steer with (see features.py).

The domains mirror the 12 TOPIC_HINTS already used in ../utils.py so results map
straight back onto the existing generation pipeline.
"""

from __future__ import annotations

# domain_key -> {"label", "probes": [...]}
DOMAINS: dict[str, dict] = {
    "vaccination": {
        "label": "HPV prevention and vaccine schedules",
        "probes": [
            "The HPV vaccine is recommended for preteens at ages 11 to 12.",
            "Gardasil 9 protects against nine HPV types and is given in two or three doses.",
            "Children who start the vaccine series before age 15 need only two doses.",
            "Vaccination before sexual debut gives the strongest protection against HPV.",
        ],
    },
    "transmission": {
        "label": "HPV transmission routes and risk factors",
        "probes": [
            "HPV spreads through skin-to-skin contact during vaginal, anal, or oral sex.",
            "A person can transmit HPV even when they have no visible symptoms.",
            "Having more sexual partners increases the risk of acquiring HPV.",
            "HPV can rarely pass from a mother to her baby during childbirth.",
        ],
    },
    "symptoms_warts": {
        "label": "HPV symptoms, genital warts, and diagnosis",
        "probes": [
            "Low-risk HPV types cause genital warts, which are soft flesh-colored bumps.",
            "Most people infected with HPV never develop any symptoms.",
            "Genital warts can appear on the genitals, anus, or throat.",
            "High-risk HPV usually causes no symptoms until precancer develops.",
        ],
    },
    "screening": {
        "label": "Cervical cancer screening (Pap smear and HPV test)",
        "probes": [
            "A Pap test collects cervical cells to look for precancerous changes.",
            "The HPV test checks for high-risk HPV types in cervical cells.",
            "Women aged 30 to 65 can have co-testing every five years.",
            "Regular screening detects precancers before they become invasive cancer.",
        ],
    },
    "cancers": {
        "label": "HPV-related cancers: cervical, anal, oropharyngeal",
        "probes": [
            "HPV 16 and 18 cause about seventy percent of cervical cancers.",
            "HPV can cause oropharyngeal cancers of the throat, tongue, and tonsils.",
            "Persistent high-risk HPV infection can progress to invasive cancer.",
            "Anal and penile cancers are linked to high-risk HPV types.",
        ],
    },
    "men": {
        "label": "HPV in men and gender-specific considerations",
        "probes": [
            "HPV can cause penile, anal, and throat cancers in men.",
            "Men who have sex with men are at higher risk for anal HPV.",
            "There is no approved routine HPV screening test for men.",
            "The HPV vaccine protects men against genital warts and cancers.",
        ],
    },
    "pregnancy": {
        "label": "HPV and pregnancy or fertility",
        "probes": [
            "HPV infection usually does not affect a woman's ability to get pregnant.",
            "Genital warts can grow faster during pregnancy due to hormonal changes.",
            "The HPV vaccine is not recommended during pregnancy.",
            "Cervical procedures for precancer can affect future pregnancies.",
        ],
    },
    "immunocompromised": {
        "label": "HPV in immunocompromised individuals (HIV, transplant)",
        "probes": [
            "People living with HIV clear HPV infections less easily.",
            "Immunocompromised patients need three doses of the HPV vaccine.",
            "Transplant recipients have a higher risk of HPV-related cancers.",
            "Weakened immunity allows persistent high-risk HPV infection.",
        ],
    },
    "types": {
        "label": "High-risk vs. low-risk HPV types and their consequences",
        "probes": [
            "Low-risk HPV types 6 and 11 cause most genital warts.",
            "High-risk types such as 16 and 18 can integrate into host DNA.",
            "There are more than two hundred related HPV virus types.",
            "Oncogenic HPV types drive precancerous cervical changes.",
        ],
    },
    "myths": {
        "label": "Common myths and misconceptions about HPV",
        "probes": [
            "It is a myth that only promiscuous people get HPV.",
            "Many people wrongly believe the HPV vaccine encourages risky behavior.",
            "Some incorrectly think condoms fully prevent HPV transmission.",
            "A common misconception is that HPV always causes symptoms.",
        ],
    },
    "treatment": {
        "label": "HPV treatment options and management of sequelae",
        "probes": [
            "There is no antiviral cure for the HPV infection itself.",
            "Genital warts can be removed with topical medication or cryotherapy.",
            "Cervical precancer is treated with a LEEP procedure.",
            "HPV-related cancers are treated with surgery, radiation, or chemotherapy.",
        ],
    },
    "epidemiology": {
        "label": "HPV epidemiology and public health statistics",
        "probes": [
            "HPV is the most common sexually transmitted infection in the United States.",
            "Nearly all sexually active people get HPV at some point in their lives.",
            "Vaccination programs have sharply reduced HPV infection rates in teens.",
            "Cervical cancer incidence is much higher where screening is unavailable.",
        ],
    },
}


# A neutral elicitation prompt. Steering — not the prompt — is what should make
# the model ask about a particular domain. Keeping the prompt domain-free is what
# lets us measure whether steering actually did the work.
ELICITATION_PROMPT = (
    "You are helping build a patient-education FAQ about HPV. "
    "Write ONE clear, specific question a patient might ask about HPV. "
    "Reply with only the question."
)


def background_probes(exclude: str | None = None) -> list[str]:
    """All probes from the OTHER domains — the contrast set for finding the
    features that are *specific* to a target domain (see LITERATURE_REVIEW.md §3)."""
    out: list[str] = []
    for key, d in DOMAINS.items():
        if key == exclude:
            continue
        out.extend(d["probes"])
    return out
