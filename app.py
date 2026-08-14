import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field

from src.helper import create_embeddings
from src.prompt import system_prompt


load_dotenv()

app = Flask(__name__)


# Environment


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY was not found.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found.")



# Embeddings + Pinecone


embeddings = create_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)


# OpenAI model

chat_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    max_tokens=350,
    timeout=30,
    max_retries=2,
)


# Supported-language detection


def detect_explicit_language(message: str) -> Optional[str]:
    """
    Detect a language only when there is enough evidence.

    Returns:
    - English
    - Swedish
    - Spanish
    - Arabic
    - None when the message is too ambiguous
    """

    text = message.lower().strip()

    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"

    if any(character in text for character in "åäö"):
        return "Swedish"

    if any(character in text for character in "áéíóúñü¿¡"):
        return "Spanish"

    words = set(
        re.findall(
            r"[a-zA-ZÀ-ÿ]+",
            text,
        )
    )

    english_words = {
        "i",
        "have",
        "pain",
        "fever",
        "chest",
        "breathing",
        "difficulty",
        "cough",
        "throat",
        "temperature",
        "yesterday",
        "today",
        "hello",
        "hi",
        "what",
        "should",
        "do",
        "started",
        "also",
        "hurt",
        "hurts",
    }

    swedish_words = {
        "jag",
        "har",
        "och",
        "svårt",
        "andas",
        "feber",
        "bröstsmärta",
        "hosta",
        "halsont",
        "temperatur",
        "sedan",
        "ont",
        "hej",
        "tack",
        "vad",
        "ska",
        "göra",
        "också",
    }

    spanish_words = {
        "tengo",
        "dolor",
        "fiebre",
        "respirar",
        "tos",
        "garganta",
        "desde",
        "hola",
        "gracias",
        "temperatura",
        "pecho",
        "qué",
        "debo",
        "hacer",
        "también",
        "tambien",
    }

    english_score = len(words & english_words)
    swedish_score = len(words & swedish_words)
    spanish_score = len(words & spanish_words)

    highest_score = max(
        english_score,
        swedish_score,
        spanish_score,
    )

    if highest_score == 0:
        return None

    if (
        swedish_score > english_score
        and swedish_score > spanish_score
    ):
        return "Swedish"

    if (
        spanish_score > english_score
        and spanish_score > swedish_score
    ):
        return "Spanish"

    if (
        english_score > swedish_score
        and english_score > spanish_score
    ):
        return "English"

    return None


def detect_user_language(message: str) -> str:
    """
    Detect the user's language.

    English is used only when no supported language can
    be confidently detected.
    """

    return detect_explicit_language(message) or "English"


def detect_response_language(
    user_message: str,
    chat_history: list[BaseMessage],
) -> str:
    """
    Keep language consistent for ambiguous follow-up messages.

    Example:
    Swedish conversation -> "39°C" -> answer in Swedish.
    """

    current_language = detect_explicit_language(user_message)

    if current_language:
        return current_language

    for message in reversed(chat_history):
        if not isinstance(message, HumanMessage):
            continue

        previous_language = detect_explicit_language(
            str(message.content)
        )

        if previous_language:
            return previous_language

    return "English"


# Structured symptom extraction


class SymptomDetails(BaseModel):
    """Information explicitly provided for the current medical problem."""

    symptom: Optional[str] = Field(
        default=None,
        description="Main symptom.",
    )

    body_location: Optional[str] = Field(
        default=None,
        description="Exact body location.",
    )

    duration: Optional[str] = Field(
        default=None,
        description="Symptom duration.",
    )

    temperature: Optional[str] = Field(
        default=None,
        description="Measured temperature.",
    )

    fever_duration: Optional[str] = Field(
        default=None,
        description="Fever duration.",
    )

    severity: Optional[str] = Field(
        default=None,
        description="Reported severity.",
    )

    trigger: Optional[str] = Field(
        default=None,
        description="Associated activity or event.",
    )

    injury: Optional[bool] = Field(
        default=None,
        description="Explicit injury or trauma status.",
    )

    radiation: Optional[str] = Field(
        default=None,
        description="Where pain spreads.",
    )

    numbness: Optional[bool] = None
    tingling: Optional[bool] = None
    weakness: Optional[bool] = None
    cough: Optional[bool] = None
    sore_throat: Optional[bool] = None
    breathing_difficulty: Optional[bool] = None
    rash: Optional[bool] = None
    vomiting: Optional[bool] = None
    stiff_neck: Optional[bool] = None
    confusion: Optional[bool] = None

    associated_symptoms: list[str] = Field(default_factory=list)
    warning_signs: list[str] = Field(default_factory=list)


symptom_extractor = chat_model.with_structured_output(
    SymptomDetails
)


def extract_symptom_details(
    user_message: str,
    chat_history: list[BaseMessage],
) -> SymptomDetails:
    """
    Extract only medical facts explicitly provided by the USER.

    Assistant messages are intentionally excluded so that questions
    such as "Do you have difficulty breathing?" are never mistaken
    for symptoms the user actually reported.
    """

    user_only_history = [
        message
        for message in chat_history
        if isinstance(message, HumanMessage)
    ]

    extraction_messages = [
        SystemMessage(
            content=(
                "Extract medical information for the CURRENT medical "
                "problem only.\n\n"

                "Use ONLY information explicitly stated by the USER.\n"
                "Never treat something mentioned by the assistant as "
                "a user symptom or fact.\n\n"

                "Use previous USER messages only when they belong to "
                "the same active medical problem.\n\n"

                "For boolean fields:\n"
                "- true = the USER explicitly reported the symptom.\n"
                "- false = the USER explicitly denied the symptom.\n"
                "- null = the USER did not mention it.\n\n"

                "Example:\n"
                "Assistant asks: 'Do you have difficulty breathing?'\n"
                "User says: 'I have a cough and sore throat.'\n"
                "Then breathing_difficulty MUST remain null.\n\n"

                "Never guess missing information. "
                "Never diagnose diseases."
            )
        ),
        *user_only_history,
        HumanMessage(
            content=user_message
        ),
    ]

    return symptom_extractor.invoke(
        extraction_messages
    )

# =========================================================
# New medical problem detection
# =========================================================

class ConversationState(BaseModel):
    """Whether the latest message starts a different complaint."""

    new_medical_problem: bool = Field(
        description=(
            "True only when the latest message starts a clearly "
            "different medical complaint."
        )
    )

    reason: str = Field(
        description="Short explanation for the decision."
    )


conversation_detector = chat_model.with_structured_output(
    ConversationState
)


def detect_new_medical_problem(
    user_message: str,
    chat_history: list[BaseMessage],
) -> ConversationState:

    if not chat_history:
        return ConversationState(
            new_medical_problem=False,
            reason="There is no previous medical problem.",
        )

    detector_messages = [
        SystemMessage(
            content=(
                "Decide whether the latest user message starts a clearly "
                "different medical complaint or continues the active "
                "complaint.\n\n"

                "Related symptoms that reasonably belong to the same "
                "illness should remain the SAME problem.\n\n"

                "Examples:\n"
                "Fever -> temperature is 39.2 = SAME.\n"
                "Fever -> sore throat = SAME.\n"
                "Fever -> cough = SAME.\n"
                "Fever -> difficulty breathing = SAME and may increase risk.\n"
                "Fever -> chest pain = SAME if presented as an additional "
                "symptom of the current illness.\n"
                "Fever -> what should I do = SAME.\n"
                "Fever -> shoulder pain after lifting boxes = NEW.\n"
                "Chest pain with breathing difficulty -> 'I have a fever' "
                "as a separate complaint = NEW.\n"
                "Back pain -> numbness in the leg = SAME.\n"
                "Heavy menstrual bleeding -> ankle pain after a fall = NEW.\n\n"

                "A dangerous new symptom that develops during an active "
                "illness should normally remain part of the SAME problem "
                "so that it can be included in risk assessment.\n\n"

                "Do not start a new problem merely because a related "
                "symptom or follow-up question appears."
            )
        ),
        *chat_history,
        HumanMessage(content=user_message),
    ]

    return conversation_detector.invoke(detector_messages)


# Risk assessment


class RiskAssessment(BaseModel):
    """Basic triage-style risk assessment."""

    risk_level: str = Field(
        description="One of: low, moderate, high."
    )

    urgent: bool = Field(
        description="Whether urgent medical assessment may be needed."
    )

    reasons: list[str] = Field(default_factory=list)

    emergency_reason: Optional[str] = None


risk_detector = chat_model.with_structured_output(
    RiskAssessment
)


def assess_medical_risk(
    user_message: str,
    symptom_details: SymptomDetails,
) -> RiskAssessment:

    symptom_json = json.dumps(
        symptom_details.model_dump(exclude_none=True),
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(
    content=(
        "Perform cautious medical triage using only symptoms "
        "explicitly provided by the user.\n\n"

        "IMPORTANT:\n"
        "Do NOT classify a case as urgent merely because the user "
        "has fever, cough, sore throat, cold symptoms, or "
        "flu-like symptoms.\n\n"

        "A fever such as 39°C together with cough and sore throat, "
        "WITHOUT another emergency warning sign, should generally "
        "be classified as LOW or MODERATE risk, not HIGH risk.\n\n"

        "Chest pain ALONE is not enough to automatically return "
        "urgent=true. Evaluate the explicitly reported associated "
        "warning signs.\n\n"

        "Return HIGH risk and urgent=true when there is a clear "
        "emergency warning pattern explicitly reported, such as:\n"
        "- chest pain together with difficulty breathing\n"
        "- persistent chest pressure with shortness of breath\n"
        "- severe difficulty breathing\n"
        "- new one-sided weakness or trouble speaking\n"
        "- loss of consciousness or inability to awaken normally\n"
        "- seizure\n"
        "- severe uncontrolled bleeding\n"
        "- fever with confusion\n"
        "- fever with stiff neck\n"
        "- severe weakness or inability to stand or walk normally\n"
        "- pregnancy with severe bleeding or severe abdominal pain\n"
        "- sudden severe testicular pain\n"
        "- prolonged painful erection\n\n"

        "Return MODERATE when medical review may be appropriate "
        "but there is no clear emergency warning sign. "
        "Examples include:\n"
        "- high or persistent fever without emergency warning signs\n"
        "- symptoms that are worsening\n"
        "- persistent respiratory symptoms without severe "
        "breathing difficulty\n"
        "- isolated chest pain without an explicitly reported "
        "emergency warning pattern\n\n"

        "IMPORTANT FOR MENSTRUAL OR VAGINAL BLEEDING:\n"
        "- Heavy menstrual bleeding by itself is NOT automatically HIGH risk.\n"
        "- Bleeding that started recently is NOT automatically HIGH risk.\n"
        "- Passing blood clots by itself is NOT automatically HIGH risk.\n"
        "- Heavy bleeding together with blood clots is NOT automatically "
        "HIGH risk unless an emergency warning sign is also explicitly reported.\n"
        "- Use MODERATE risk when heavy menstrual bleeding or blood clots "
        "need medical review but no emergency warning sign has been "
        "explicitly reported.\n"
        "- Use HIGH risk and urgent=true only when an explicit emergency "
        "warning sign is reported, such as fainting, severe dizziness, "
        "confusion, severe weakness, inability to stand or walk normally, "
        "severe uncontrolled bleeding, severe abdominal or pelvic pain, "
        "or pregnancy with severe bleeding or severe abdominal pain.\n"
        "- Never assume bleeding is life-threatening from the words "
        "'heavy', 'large clots', or 'started yesterday' alone.\n"
        "- Never invent an emergency warning sign that the user did not report.\n\n"

        "Return LOW for common mild symptoms without warning signs.\n\n"

        "Never invent symptoms. "
        "Do not assume an unmentioned warning sign is present. "
        "Do not diagnose diseases."
    )
),
        HumanMessage(
            content=(
                f"User message:\n{user_message}\n\n"
                f"Extracted information:\n{symptom_json}"
            )
        ),
    ]

    return risk_detector.invoke(messages)


# Urgent response


def generate_urgent_response(
    user_message: str,
    risk_assessment: RiskAssessment,
    target_language: str,
) -> str:

    risk_json = json.dumps(
        risk_assessment.model_dump(exclude_none=True),
        ensure_ascii=False,
    )

    response = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a cautious medical information assistant.\n\n"

                    "A separate triage system has determined that the "
                    "user's symptoms may require urgent medical "
                    "assessment.\n\n"

                    f"OUTPUT LANGUAGE: {target_language}.\n"
                    f"You MUST write the entire response only in "
                    f"{target_language}.\n"
                    "Do not switch to another language.\n\n"

                    "Your response must:\n"
                    "- briefly acknowledge the symptoms\n"
                    "- clearly advise urgent medical assessment now\n"
                    "- advise contacting emergency medical services if "
                    "symptoms are severe, rapidly worsening, or the person "
                    "feels faint or critically unwell\n"
                    "- not diagnose a disease\n"
                    "- not list speculative diagnoses\n"
                    "- not delay care by asking additional questions\n"
                    "- be concise and under 120 words\n"
                )
            ),
            HumanMessage(
                content=(
                    f"User message:\n{user_message}\n\n"
                    f"Required response language:\n{target_language}\n\n"
                    f"Risk assessment:\n{risk_json}"
                )
            ),
        ]
    )

    return str(response.content).strip()


# Medical topics


TOPIC_KEYWORDS = {
    "musculoskeletal": {
        "back",
        "shoulder",
        "shoulders",
        "neck",
        "muscle",
        "lifting",
        "boxes",
        "spine",
        "arm",
        "leg",
        "strain",
        "sprain",
        "rygg",
        "axel",
        "axlar",
        "muskel",
        "lyft",
        "lådor",
        "espalda",
        "hombro",
        "músculo",
        "ظهر",
        "كتف",
        "عضلات",
    },

    "sexual_health": {
        "sex",
        "sexual",
        "libido",
        "desire",
        "erection",
        "penis",
        "testicle",
        "intercourse",
        "ejaculation",
        "sexuell",
        "lust",
        "samlag",
        "sexo",
        "deseo",
        "pene",
        "جنس",
        "رغبة",
        "قضيب",
        "انتصاب",
    },

    "menstrual_health": {
        "period",
        "menstrual",
        "menstruation",
        "bleeding",
        "clots",
        "vaginal",
        "mens",
        "blödning",
        "periodo",
        "menstruación",
        "sangrado",
        "دورة",
        "حيض",
        "نزيف",
    },

    "fever": {
        "fever",
        "temperature",
        "chills",
        "feber",
        "temperatur",
        "fiebre",
        "حرارة",
        "حمى",
    },

    "cardiopulmonary": {
        "chest pain",
        "chest pressure",
        "chest",
        "difficulty breathing",
        "shortness of breath",
        "bröstsmärta",
        "bröst",
        "svårt att andas",
        "dolor en el pecho",
        "pecho",
        "dificultad para respirar",
        "ألم في الصدر",
        "الصدر",
        "صعوبة في التنفس",
    },

    "respiratory": {
        "cough",
        "breathing",
        "wheezing",
        "sore throat",
        "cold",
        "flu",
        "hosta",
        "andning",
        "halsont",
        "tos",
        "respirar",
        "سعال",
        "تنفس",
    },

    "urinary": {
        "urine",
        "urination",
        "pee",
        "bladder",
        "kidney",
        "urin",
        "kissa",
        "njure",
        "orina",
        "riñón",
        "بول",
        "تبول",
        "كلية",
    },
}


CATEGORY_TERMS = {
    "musculoskeletal": {
        "back",
        "shoulder",
        "muscle",
        "joint",
        "bone",
        "spine",
        "injury",
        "strain",
        "sprain",
        "orthopedic",
    },

    "sexual_health": {
        "sexual",
        "sex",
        "penis",
        "erectile",
        "libido",
        "testicular",
        "reproductive",
        "relationship",
        "ejaculation",
    },

    "menstrual_health": {
        "menstrual",
        "menstruation",
        "period",
        "vaginal",
        "uterine",
        "ovarian",
        "women",
        "bleeding",
    },

    "fever": {
        "fever",
        "temperature",
        "infection",
    },

    "cardiopulmonary": {
        "chest",
        "cardiac",
        "heart",
        "breathing",
        "respiratory",
        "lung",
        "shortness of breath",
    },

    "respiratory": {
        "cough",
        "lung",
        "respiratory",
        "breathing",
        "asthma",
        "cold",
        "influenza",
        "throat",
    },

    "urinary": {
        "urinary",
        "urination",
        "urine",
        "bladder",
        "kidney",
    },
}


PREFERRED_TITLES = {
    "fever": {
        "fever",
        "high fever",
        "viral infections",
        "influenza",
        "common cold",
        "sore throat",
    },

    "respiratory": {
        "common cold",
        "influenza",
        "sore throat",
        "cough",
        "viral infections",
    },

    "sexual_health": {
        "erectile dysfunction",
        "weak erection",
        "sexual problems in men",
        "erection problems",
    },

    "musculoskeletal": {
        "muscle strain",
        "shoulder pain",
        "back pain",
        "neck pain",
        "pain between shoulder blades",
    },
}

def detect_topic(
    query: str,
) -> Optional[str]:

    normalized_query = " ".join(
        query.lower().split()
    )

    best_topic = None
    best_matches = 0

    for topic, keywords in TOPIC_KEYWORDS.items():

        matches = sum(
            1
            for keyword in keywords
            if keyword in normalized_query
        )

        if matches > best_matches:
            best_topic = topic
            best_matches = matches

    return best_topic

# Deterministic medical-problem switch protection


def should_force_new_problem(
    user_message: str,
    chat_history: list[BaseMessage],
) -> bool:
    """
    Detect an obvious switch between clearly different medical problems.

    The LLM detector still handles ambiguous cases.
    """

    if not chat_history:
        return False

    normalized_message = " ".join(
        user_message.lower()
        .strip(" !?.,")
        .split()
    )

    continuation_markers = {
        "also",
        "still",
        "and now",
        "it also",
        "i also",
        "too",
        "också",
        "fortfarande",
        "jag har också",
        "también",
        "todavía",
        "tambien",
        "أيضا",
        "أيضاً",
        "كمان",
    }

    has_continuation_marker = any(
        marker in normalized_message
        for marker in continuation_markers
    )

    current_topic = detect_topic(
        user_message
    )

    # Negative symptom statements normally belong to
    # the active medical problem and should not create
    # a new complaint.
    negative_followup_patterns = {
        "i don't have",
        "i do not have",
        "i haven't had",
        "i have no",
        "no bleeding",
        "without bleeding",
        "jag har inte",
        "jag har ingen",
        "jag har inget",
        "no tengo",
        "sin sangrado",
        "ليس لدي",
        "لا أعاني",
        "ما عندي",
    }

    if any(
        pattern in normalized_message
        for pattern in negative_followup_patterns
    ):
        return False

    if current_topic is None:
        # Example: "What should I do?"
        return False

    previous_topic = None

    for message in reversed(
        chat_history
    ):
        if not isinstance(
            message,
            HumanMessage,
        ):
            continue

        candidate_topic = detect_topic(
            str(message.content)
        )

        if candidate_topic:
            previous_topic = candidate_topic
            break

    if previous_topic is None:
        return False

    if current_topic == previous_topic:
        return False

    # Fever and respiratory symptoms commonly belong
    # to the same illness.
    if (
        previous_topic in {"fever", "respiratory"}
        and current_topic in {
            "fever",
            "respiratory",
            "cardiopulmonary",
        }
    ):
        return False

    # If an additional symptom is explicitly added, allow
    # the LLM detector to decide whether it is still the same case.
    if has_continuation_marker:
        return False

    # Explicit fresh fever/respiratory complaint after a
    # cardiopulmonary case should reset the active case.
    if (
        previous_topic == "cardiopulmonary"
        and current_topic in {
            "fever",
            "respiratory",
        }
    ):
        return True

    return True



# Missing information


def get_missing_information(
    details: SymptomDetails,
) -> list[str]:

    missing = []

    symptom = (
        details.symptom or ""
    ).lower()

    location = (
        details.body_location or ""
    ).lower()

    if (
        "fever" in symptom
        or details.temperature is not None
        or details.fever_duration is not None
    ):

        if details.temperature is None:
            missing.append(
                "measured temperature"
            )

        if (
            details.fever_duration is None
            and details.duration is None
        ):
            missing.append(
                "fever duration"
            )

        if details.cough is None:
            missing.append(
                "cough"
            )

        if details.sore_throat is None:
            missing.append(
                "sore throat"
            )

        if details.breathing_difficulty is None:
            missing.append(
                "breathing difficulty"
            )

        return missing[:4]

    musculoskeletal_words = {
        "back",
        "shoulder",
        "neck",
        "arm",
        "leg",
        "spine",
        "knee",
        "hip",
        "ankle",
        "foot",
    }

    if (
        "pain" in symptom
        and any(
            word in location
            for word in musculoskeletal_words
        )
    ):

        if details.severity is None:
            missing.append(
                "pain severity"
            )

        if details.radiation is None:
            missing.append(
                "whether the pain spreads"
            )

        if details.numbness is None:
            missing.append(
                "numbness"
            )

        if details.weakness is None:
            missing.append(
                "weakness"
            )

        return missing[:4]

    genital_words = {
        "penis",
        "testicle",
        "scrotum",
        "genital",
    }

    if any(
        word in location
        for word in genital_words
    ):

        if details.severity is None:
            missing.append(
                "pain severity"
            )

        if details.duration is None:
            missing.append(
                "duration"
            )

        return missing[:4]

    return missing


# Case-aware retrieval query


def build_retrieval_query(
    user_message: str,
    details: SymptomDetails,
) -> str:

    parts = []

    if details.symptom:
        parts.append(
            details.symptom
        )

    if details.body_location:
        parts.append(
            f"location: {details.body_location}"
        )

    if details.temperature:
        parts.append(
            f"temperature: {details.temperature}"
        )

    duration = (
        details.fever_duration
        or details.duration
    )

    if duration:
        parts.append(
            f"duration: {duration}"
        )

    if details.severity:
        parts.append(
            f"severity: {details.severity}"
        )

    if details.trigger:
        parts.append(
            f"trigger: {details.trigger}"
        )

    if details.radiation:
        parts.append(
            f"radiation: {details.radiation}"
        )

    boolean_labels = {
        "cough": "cough",
        "sore_throat": "sore throat",
        "breathing_difficulty": "difficulty breathing",
        "rash": "rash",
        "vomiting": "vomiting",
        "stiff_neck": "stiff neck",
        "confusion": "confusion",
        "numbness": "numbness",
        "tingling": "tingling",
        "weakness": "weakness",
        "injury": "injury or trauma",
    }

    for field_name, label in boolean_labels.items():

        if getattr(
            details,
            field_name
        ) is True:

            parts.append(
                label
            )

    parts.extend(
        details.associated_symptoms
    )

    parts.extend(
        details.warning_signs
    )

    unique_parts = dict.fromkeys(
        part.strip()
        for part in parts
        if part and part.strip()
    )

    case_summary = "; ".join(
        unique_parts
    )

    if case_summary:
        return (
            f"Current medical problem: {case_summary}. "
            f"User asks: {user_message}"
        )

    return user_message


# English retrieval query


def translate_query_for_retrieval(
    query: str,
) -> str:

    response = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the text as a concise standalone English "
                    "medical search query for retrieval from a medical "
                    "database.\n\n"

                    "Translate into English when needed.\n"
                    "Keep all medically important symptoms, location, "
                    "duration, temperature, severity, triggers, injury "
                    "and warning signs.\n\n"

                    "If the text contains a follow-up request such as "
                    "'what should I do', preserve the medical case "
                    "details and remove only conversational filler.\n\n"

                    "Do not add diseases or diagnoses that the user did "
                    "not mention.\n"
                    "Do not answer the question.\n"
                    "Return only the search query."
                )
            ),

            HumanMessage(
                content=query
            ),
        ]
    )

    english_query = str(
        response.content
    ).strip()

    return (
        english_query
        or query
    )


# Retrieval + reranking


def retrieve_and_rerank(
    query: str,
):

    english_query = translate_query_for_retrieval(
        query
    )

    detected_topic = detect_topic(
        english_query
    )

    app.logger.info(
        "Retrieval | original=%r | english=%r | topic=%r",
        query,
        english_query,
        detected_topic,
    )

    results = docsearch.similarity_search_with_relevance_scores(
        query=english_query,
        k=30,
        score_threshold=0.35,
    )

    topic_terms = CATEGORY_TERMS.get(
        detected_topic,
        set(),
    )

    preferred_titles = PREFERRED_TITLES.get(
        detected_topic,
        set(),
    )

    normalized_query = english_query.lower()

    query_words = {
        word
        for word in normalized_query
        .replace("?", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .split()
        if len(word) >= 4
    }

    
    # Query flags
  

    query_has_fever = any(
        term in normalized_query
        for term in {
            "fever",
            "high fever",
            "temperature",
            "38°",
            "39°",
            "40°",
        }
    )

    trauma_terms = {
        "fall",
        "accident",
        "collision",
        "trauma",
        "dislocation",
        "broken",
        "fracture",
    }

    serious_injury_terms = {
        "dislocated",
        "dislocation",
        "fracture",
        "broken bone",
        "major trauma",
    }

    query_mentions_trauma = any(
        term in normalized_query
        for term in trauma_terms
    )

    heat_exposure_terms = {
        "heat",
        "hot weather",
        "sun",
        "exercise in heat",
        "overheating",
    }

    query_mentions_heat = any(
        term in normalized_query
        for term in heat_exposure_terms
    )

    query_has_erection_problem = any(
        term in normalized_query
        for term in {
            "erection",
            "erectile",
            "erectile dysfunction",
            "weak erection",
            "trouble getting an erection",
            "difficulty getting an erection",
            "difficulty maintaining an erection",
        }
    )

    query_mentions_ejaculation = any(
        term in normalized_query
        for term in {
            "ejaculation",
            "premature ejaculation",
            "ejaculate",
        }
    )

    query_mentions_erection_pain = any(
        term in normalized_query
        for term in {
            "painful erection",
            "pain during erection",
            "erection pain",
        }
    )

    # Rerank documents
    

    reranked_results = []

    for document, vector_score in results:

        metadata = document.metadata or {}

        title = str(
            metadata.get(
                "title",
                "",
            )
        ).lower()

        searchable_metadata = " ".join(
            [
                str(metadata.get("title", "")),
                str(metadata.get("category", "")),
                str(metadata.get("categories", "")),
                str(metadata.get("mapped_title", "")),
                str(metadata.get("document_type", "")),
            ]
        ).lower()

        title_overlap = sum(
            1
            for word in query_words
            if word in title
        )

        category_overlap = sum(
            1
            for term in topic_terms
            if term in searchable_metadata
        )

        curated_bonus = (
            0.10
            if metadata.get("document_type") == "curated_topic"
            else 0.0
        )

        exact_topic_bonus = 0.0
        mismatch_penalty = 0.0

        if detected_topic:

            for term in topic_terms:

                if term in title:
                    exact_topic_bonus += 0.10

            exact_topic_bonus = min(
                exact_topic_bonus,
                0.30,
            )

        if title in preferred_titles:
            exact_topic_bonus += 0.45

        # Fever-specific ranking
        

        if detected_topic == "fever" or query_has_fever:

            if title == "fever":
                exact_topic_bonus += 0.55

            elif title == "high fever":
                exact_topic_bonus += 0.25

            unrelated_fever_titles = {
                "post-covid conditions",
                "long covid",
                "headache",
                "lyme disease",
                "tick bites",
                "mosquito bites",
                "staphylococcal infections",
                "cellulitis",
                "valley fever",
                "yellow fever",
                "hemorrhagic fever",
                "hemorrhagic fevers",
                "dengue",
                "malaria",
                "heat stroke",
                "heat exhaustion",
                "heat illness",
                "hay fever",
                "fifth disease",
                "meningococcal disease",
                "chickenpox",
                "polio and post-polio syndrome",
                "plague",
                "low-grade fever",
                "bird flu",
                "avian influenza",
                "children's health",
                "histoplasmosis",
                "diphtheria",
                "pneumonia",
                "anaphylaxis",
                "vital signs",
            }

            if any(
                unrelated_title in title
                and unrelated_title not in normalized_query
                for unrelated_title in unrelated_fever_titles
            ):
                mismatch_penalty += 0.85

            if (
                "heat stroke" in title
                and not query_mentions_heat
            ):
                mismatch_penalty += 0.55

            if (
                "heat exhaustion" in title
                and not query_mentions_heat
            ):
                mismatch_penalty += 0.55

            if (
                "heat illness" in title
                and not query_mentions_heat
            ):
                mismatch_penalty += 0.55

       
        # Respiratory-specific ranking
      

        if detected_topic == "respiratory":

            if title in {
                "common cold",
                "influenza",
                "sore throat",
                "cough",
                "viral infections",
                "fever",
            }:
                exact_topic_bonus += 0.20

            disease_specific_respiratory_titles = {
                "respiratory syncytial virus infections",
                "sinusitis",
                "bronchitis",
                "pneumonia",
                "bird flu",
                "avian influenza",
                "histoplasmosis",
                "diphtheria",
                "anaphylaxis",
            }

            if any(
                disease_title in title
                and disease_title not in normalized_query
                for disease_title in disease_specific_respiratory_titles
            ):
                mismatch_penalty += 0.55

        # Sexual-health-specific ranking
        

        if detected_topic == "sexual_health":

            if query_has_erection_problem:

                if title in {
                    "erectile dysfunction",
                    "weak erection",
                    "erection problems",
                    "sexual problems in men",
                }:
                    exact_topic_bonus += 0.50

                if (
                    "premature ejaculation" in title
                    and not query_mentions_ejaculation
                ):
                    mismatch_penalty += 0.75

                if (
                    "pain during erection" in title
                    and not query_mentions_erection_pain
                ):
                    mismatch_penalty += 0.75

                if "sexual problems in women" in title:
                    mismatch_penalty += 0.90

                if "birth control" in title:
                    mismatch_penalty += 0.90

            if query_mentions_ejaculation:

                if "premature ejaculation" in title:
                    exact_topic_bonus += 0.55

            if query_mentions_erection_pain:

                if "pain during erection" in title:
                    exact_topic_bonus += 0.55

        
        # Musculoskeletal-specific ranking
       

        if detected_topic == "musculoskeletal":

            unrelated_injury_titles = {
                "dislocated shoulder",
                "shoulder dislocation",
                "fracture",
                "broken bone",
            }

            if (
                not query_mentions_trauma
                and any(
                    injury_title in title
                    for injury_title in unrelated_injury_titles
                )
            ):
                mismatch_penalty += 0.45

        
        # Generic trauma mismatch
        

        if (
            not query_mentions_trauma
            and any(
                term in searchable_metadata
                for term in serious_injury_terms
            )
        ):
            mismatch_penalty += 0.22

        final_score = (
            float(vector_score)
            + (title_overlap * 0.12)
            + (category_overlap * 0.08)
            + curated_bonus
            + exact_topic_bonus
            - mismatch_penalty
        )

        reranked_results.append(
            (
                document,
                final_score,
            )
        )

    reranked_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    
    # Generic fever source filtering
  

    generic_fever_sources = {
        "fever",
        "high fever",
        "influenza",
        "common cold",
        "viral infections",
        "sore throat",
        "cough",
    }

    specific_fever_topics = {
        "pneumonia",
        "anaphylaxis",
        "histoplasmosis",
        "diphtheria",
        "bird flu",
        "avian influenza",
        "lyme disease",
        "malaria",
        "dengue",
        "yellow fever",
        "valley fever",
        "hay fever",
        "heat stroke",
        "heat exhaustion",
        "heat illness",
        "plague",
    }

    generic_fever_case = (
        query_has_fever
        and not any(
            specific_term in normalized_query
            for specific_term in specific_fever_topics
        )
    )

    
    # Generic erection-problem source filtering
    

    generic_erection_sources = {
        "erectile dysfunction",
        "weak erection",
        "erection problems",
        "sexual problems in men",
    }

    generic_erection_case = (
        query_has_erection_problem
        and not query_mentions_ejaculation
        and not query_mentions_erection_pain
    )

   
    # Select final documents
   

    selected_documents = []
    seen_sources = set()

    for document, final_score in reranked_results:

        metadata = document.metadata or {}

        title = str(
            metadata.get(
                "title",
                "",
            )
        ).lower()

        # directly relevant fever/respiratory sources.
        if (
            generic_fever_case
            and title not in generic_fever_sources
        ):
            continue

        # questions, should use male erection-related sources.
        if (
            generic_erection_case
            and title not in generic_erection_sources
        ):
            continue

        unique_key = (
            metadata.get("source_url")
            or metadata.get("title")
            or document.page_content[:100]
        )

        if unique_key in seen_sources:
            continue

        if final_score < 0.65:
            continue

        seen_sources.add(
            unique_key
        )

        selected_documents.append(
            document
        )

        if len(selected_documents) == 4:
            break

    app.logger.info(
        "Selected sources: %s",
        [
            document.metadata.get("title")
            for document in selected_documents
        ],
    )

    return selected_documents
def retrieve_from_chain_inputs(
    inputs: dict,
):

    retrieval_query = str(
        inputs.get(
            "retrieval_query"
        )
        or inputs.get(
            "input"
        )
        or ""
    ).strip()

    return retrieve_and_rerank(
        retrieval_query
    )


retriever = RunnableLambda(
    retrieve_from_chain_inputs
)


# RAG chain

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),

        (
            "system",
            (
                "Structured information for the CURRENT "
                "medical problem:\n"
                "{symptom_details}\n\n"

                "Important missing information:\n"
                "{missing_information}\n\n"

                "REQUIRED OUTPUT LANGUAGE:\n"
                "{response_language}\n\n"

                "Rules:\n"
                "- Write the entire response only in "
                "{response_language}.\n"

                "- Do not switch to another language.\n"

                "- Do not ask for information already present in "
                "symptom_details.\n"

                "- If clarification is needed, ask at most two "
                "important missing items.\n"

                "- Do not mix symptoms from an older unrelated "
                "problem into the current problem.\n"

                "- If enough information is available, give practical "
                "guidance instead of asking unnecessary questions.\n"

                "- Ground medical claims in the retrieved context.\n"

                "- Do not invent a diagnosis.\n"

                "- Do not name speculative possible diseases merely "
                "because they could cause the symptoms.\n"

                "- Unless the user specifically asks about possible "
                "causes or diagnoses, prefer broad wording such as "
                "'a respiratory infection' instead of listing diseases "
                "such as pneumonia, bronchitis, or other conditions.\n"

                "- Never claim that the user has a disease that has not "
                "been diagnosed."
            ),
        ),

        MessagesPlaceholder(
            "chat_history"
        ),

        (
            "human",
            "{input}",
        ),
    ]
)


question_answer_chain = create_stuff_documents_chain(
    chat_model,
    answer_prompt,
)


rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


# Chat history


def convert_chat_history(
    history_data,
) -> list[BaseMessage]:

    messages = []

    if not isinstance(
        history_data,
        list,
    ):
        return messages

    for item in history_data[-10:]:

        if not isinstance(
            item,
            dict,
        ):
            continue

        role = item.get(
            "role"
        )

        content = str(
            item.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        if role == "user":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":

            messages.append(
                AIMessage(
                    content=content
                )
            )

    return messages



# Casual conversation


CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "see you",
    "ok",
    "okay",
    "great",
    "nice",
    "awesome",
    "how are you",
    "can you help me",
    "i need help",
    "i want help",
    "i have a question",
    "i want to ask a question",
    "can i ask a question",
    "can i ask something",
    "i want to ask something",
    "help me",

    "hej",
    "hallå",
    "god morgon",
    "god kväll",
    "tack",
    "tack så mycket",
    "hej då",
    "okej",
    "hur mår du",
    "kan du hjälpa mig",
    "jag behöver hjälp",
    "jag har en fråga",
    "jag vill ställa en fråga",

    "hola",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "gracias",
    "adiós",
    "adios",
    "hasta luego",
    "puedes ayudarme",
    "necesito ayuda",
    "tengo una pregunta",
    "quiero hacer una pregunta",

    "مرحبا",
    "مرحباً",
    "السلام عليكم",
    "وعليكم السلام",
    "شكرا",
    "شكراً",
    "مع السلامة",
    "هل يمكنك مساعدتي",
    "أحتاج مساعدة",
    "لدي سؤال",
    "أريد أن أسأل سؤالاً",
}


MEDICAL_HINTS = {
    "pain",
    "fever",
    "bleeding",
    "cough",
    "headache",
    "dizzy",
    "dizziness",
    "vomit",
    "vomiting",
    "diarrhea",
    "rash",
    "swelling",
    "breathing",
    "penis",
    "period",
    "pregnant",
    "pregnancy",
    "urine",
    "back",
    "shoulder",
    "chest",
    "stomach",
    "sex",
    "libido",

    "ont",
    "feber",
    "blödning",
    "hosta",
    "rygg",
    "axel",
    "sexuell",
    "bröst",
    "andas",

    "dolor",
    "fiebre",
    "sangrado",
    "tos",
    "sexo",
    "pecho",
    "respirar",

    "ألم",
    "حرارة",
    "حمى",
    "نزيف",
    "سعال",
    "جنس",
    "صدر",
    "تنفس",
}


def normalize_message(
    message: str,
) -> str:

    return " ".join(
        message.lower()
        .strip(" !?.,")
        .split()
    )


def is_casual_message(
    message: str,
) -> bool:

    normalized = normalize_message(
        message
    )

    if any(
        medical_hint in normalized
        for medical_hint in MEDICAL_HINTS
    ):
        return False

    if normalized in CASUAL_MESSAGES:
        return True

    return any(
        phrase in normalized
        for phrase in CASUAL_MESSAGES
        if len(
            phrase.split()
        ) >= 3
    )


def get_casual_response(
    message: str,
) -> str:

    normalized = normalize_message(
        message
    )

    language = detect_user_language(
        message
    )

    if language == "Arabic":

        if "شك" in normalized:
            return (
                "على الرحب والسعة! "
                "يمكنك طرح أي سؤال طبي."
            )

        if "مع السلامة" in normalized:
            return (
                "مع السلامة! "
                "أتمنى لك الصحة والعافية."
            )

        return (
            "مرحباً! كيف يمكنني مساعدتك "
            "في سؤالك الطبي؟"
        )

    if language == "Spanish":

        if "gracias" in normalized:
            return (
                "¡De nada! Puedes hacerme "
                "cualquier pregunta médica."
            )

        if (
            "adiós" in normalized
            or "adios" in normalized
        ):
            return (
                "¡Adiós! Cuídate."
            )

        return (
            "¡Hola! ¿Cómo puedo ayudarte "
            "con tu pregunta médica?"
        )

    if language == "Swedish":

        if "tack" in normalized:
            return (
                "Varsågod! Du kan ställa "
                "vilken medicinsk fråga du vill."
            )

        if "hej då" in normalized:
            return (
                "Hej då! Ta hand om dig."
            )

        return (
            "Hej! Hur kan jag hjälpa dig "
            "med din medicinska fråga?"
        )

    if (
        "thank" in normalized
        or "thanks" in normalized
    ):
        return (
            "You're welcome! You can ask me "
            "any medical question."
        )

    if (
        "bye" in normalized
        or "goodbye" in normalized
    ):
        return (
            "Goodbye! Take care."
        )

    return (
        "Hello! How can I help you "
        "with your medical question?"
    )


def history_has_medical_user_message(
    chat_history: list[BaseMessage],
) -> bool:

    for message in chat_history:

        if not isinstance(
            message,
            HumanMessage,
        ):
            continue

        normalized = normalize_message(
            str(
                message.content
            )
        )

        if any(
            hint in normalized
            for hint in MEDICAL_HINTS
        ):
            return True

    return False


# Flask routes


@app.route("/")
def index():

    return render_template(
        "chat.html"
    )


@app.route(
    "/get",
    methods=["POST"],
)
def chat():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    user_message = str(
        data.get(
            "msg",
            "",
        )
    ).strip()

    history_data = data.get(
        "history",
        [],
    )

    if not user_message:

        return jsonify(
            {
                "error": (
                    "Please enter a medical question."
                )
            }
        ), 400

    if is_casual_message(
        user_message
    ):

        return jsonify(
            {
                "answer": get_casual_response(
                    user_message
                ),
                "sources": [],
                "reset_history": False,
                "urgent": False,
            }
        )

    try:

        full_chat_history = convert_chat_history(
            history_data
        )

        
        # Detect obvious medical topic switches first
       

        forced_new_problem = should_force_new_problem(
            user_message=user_message,
            chat_history=full_chat_history,
        )

        if forced_new_problem:

            conversation_state = ConversationState(
                new_medical_problem=True,
                reason=(
                    "The latest message clearly starts a different "
                    "medical topic from the active complaint."
                ),
            )

        elif history_has_medical_user_message(
            full_chat_history
        ):

            normalized_current_message = normalize_message(
                user_message
            )

            negative_followup_patterns = {
                "i don't have",
                "i do not have",
                "i haven't had",
                "i have no",
                "no bleeding",
                "without bleeding",
                "jag har inte",
                "jag har ingen",
                "jag har inget",
                "no tengo",
                "sin sangrado",
                "ليس لدي",
                "لا أعاني",
                "ما عندي",
            }

            if any(
                pattern in normalized_current_message
                for pattern in negative_followup_patterns
            ):
                conversation_state = ConversationState(
                    new_medical_problem=False,
                    reason=(
                        "The latest message is a negative symptom "
                        "follow-up for the active medical problem."
                    ),
                )

            else:
                conversation_state = detect_new_medical_problem(
                    user_message=user_message,
                    chat_history=full_chat_history,
                )

        else:

            conversation_state = ConversationState(
                new_medical_problem=False,
                reason=(
                    "No previous medical complaint exists."
                ),
            )

        app.logger.info(
            "Conversation state: %s",
            conversation_state.model_dump(),
        )

        active_chat_history = (
            []
            if conversation_state.new_medical_problem
            else full_chat_history
        )

        symptom_details = extract_symptom_details(
            user_message=user_message,
            chat_history=active_chat_history,
        )

        symptom_data = symptom_details.model_dump(
            exclude_none=True
        )

        app.logger.info(
            "Extracted symptoms: %s",
            symptom_data,
        )

        risk_assessment = assess_medical_risk(
            user_message=user_message,
            symptom_details=symptom_details,
        )

        app.logger.info(
            "Risk assessment: %s",
            risk_assessment.model_dump(),
        )

        response_language = detect_response_language(
            user_message=user_message,
            chat_history=active_chat_history,
        )

        app.logger.info(
            "Response language: %s",
            response_language,
        )

        
        # Urgent cases
        

        if risk_assessment.urgent:

            urgent_answer = generate_urgent_response(
                user_message=user_message,
                risk_assessment=risk_assessment,
                target_language=response_language,
            )

            return jsonify(
                {
                    "answer": urgent_answer,
                    "sources": [],
                    "urgent": True,
                    "risk_level": (
                        risk_assessment.risk_level
                    ),
                    "risk_reasons": (
                        risk_assessment.reasons
                    ),
                    "reset_history": (
                        conversation_state
                        .new_medical_problem
                    ),
                }
            )

        
        # Normal medical conversation
        

        missing_information = get_missing_information(
            symptom_details
        )

        symptom_context = json.dumps(
            symptom_data,
            ensure_ascii=False,
            indent=2,
        )

        missing_context = json.dumps(
            missing_information,
            ensure_ascii=False,
        )

        retrieval_query = build_retrieval_query(
            user_message=user_message,
            details=symptom_details,
        )

        app.logger.info(
            "Missing information: %s",
            missing_information,
        )

        app.logger.info(
            "Case-aware retrieval query: %s",
            retrieval_query,
        )

        response = rag_chain.invoke(
            {
                "input": user_message,
                "retrieval_query": retrieval_query,
                "chat_history": active_chat_history,
                "symptom_details": symptom_context,
                "missing_information": missing_context,
                "response_language": response_language,
            }
        )

        answer = str(
            response.get(
                "answer",
                "",
            )
        ).strip()

        retrieved_documents = response.get(
            "context",
            [],
        )

        no_relevant_information = (
            "I could not find enough relevant information "
            "in the medical knowledge base."
        )

        if (
            no_relevant_information.lower()
            in answer.lower()
        ):
            retrieved_documents = []

        sources = []
        seen_sources = set()

        for document in retrieved_documents:

            metadata = (
                document.metadata
                or {}
            )

            title = metadata.get(
                "title",
                "Medical topic",
            )

            source = metadata.get(
                "source",
                "MedlinePlus",
            )

            source_url = metadata.get(
                "source_url",
                "",
            )

            unique_key = (
                source_url
                or title
            )

            if unique_key in seen_sources:
                continue

            seen_sources.add(
                unique_key
            )

            sources.append(
                {
                    "title": title,
                    "source": source,
                    "url": source_url,
                }
            )

        if not answer:

            return jsonify(
                {
                    "error": (
                        "I could not generate an answer."
                    )
                }
            ), 500

        return jsonify(
            {
                "answer": answer,
                "sources": sources[:3],
                "urgent": False,
                "risk_level": (
                    risk_assessment.risk_level
                ),
                "reset_history": (
                    conversation_state
                    .new_medical_problem
                ),
            }
        )

    except Exception:

        app.logger.exception(
            "Chatbot error"
        )

        return jsonify(
            {
                "error": (
                    "The medical assistant is temporarily "
                    "unavailable. Please try again."
                )
            }
        ), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "false",
            ).lower()
            == "true"
        ),
    )