import json
import os
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
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
from src.prompt import (
    contextualize_q_system_prompt,
    system_prompt,
)


load_dotenv()

app = Flask(__name__)



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY was not found."
    )

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found."
    )



# Embeddings + Pinecone


embeddings = create_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)


# OpenAI model


chat_model = ChatOpenAI(
    model=os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    ),
    temperature=0,
    max_tokens=350,
    timeout=30,
    max_retries=2,
)

# Structured symptom extraction

class SymptomDetails(BaseModel):
    """Information explicitly provided for the current medical problem."""

    symptom: Optional[str] = Field(
        default=None,
        description=(
            "Main symptom such as pain, fever, cough, or bleeding."
        ),
    )

    body_location: Optional[str] = Field(
        default=None,
        description=(
            "Exact body location explicitly reported."
        ),
    )

    duration: Optional[str] = Field(
        default=None,
        description=(
            "How long the current symptom has been present."
        ),
    )

    temperature: Optional[str] = Field(
        default=None,
        description=(
            "Measured body temperature explicitly provided."
        ),
    )

    fever_duration: Optional[str] = Field(
        default=None,
        description=(
            "How long the fever has lasted."
        ),
    )

    severity: Optional[str] = Field(
        default=None,
        description=(
            "Reported severity such as mild, moderate, or severe."
        ),
    )

    trigger: Optional[str] = Field(
        default=None,
        description=(
            "Activity or event associated with the symptom."
        ),
    )

    injury: Optional[bool] = Field(
        default=None,
        description=(
            "True only if injury or trauma was explicitly reported. "
            "False only if explicitly denied. Otherwise null."
        ),
    )

    radiation: Optional[str] = Field(
        default=None,
        description=(
            "Where the pain spreads, if explicitly reported."
        ),
    )

    numbness: Optional[bool] = Field(
        default=None,
        description=(
            "True if numbness was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    tingling: Optional[bool] = Field(
        default=None,
        description=(
            "True if tingling was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    weakness: Optional[bool] = Field(
        default=None,
        description=(
            "True if weakness was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    cough: Optional[bool] = Field(
        default=None,
        description=(
            "True if cough was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    sore_throat: Optional[bool] = Field(
        default=None,
        description=(
            "True if sore throat was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    breathing_difficulty: Optional[bool] = Field(
        default=None,
        description=(
            "True if breathing difficulty was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    rash: Optional[bool] = Field(
        default=None,
        description=(
            "True if rash was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    vomiting: Optional[bool] = Field(
        default=None,
        description=(
            "True if vomiting was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    stiff_neck: Optional[bool] = Field(
        default=None,
        description=(
            "True if neck stiffness was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    confusion: Optional[bool] = Field(
        default=None,
        description=(
            "True if confusion was explicitly reported. "
            "False if explicitly denied. Otherwise null."
        ),
    )

    associated_symptoms: list[str] = Field(
        default_factory=list,
        description=(
            "Other symptoms explicitly reported for this problem."
        ),
    )

    warning_signs: list[str] = Field(
        default_factory=list,
        description=(
            "Serious warning signs explicitly reported."
        ),
    )


symptom_extractor = chat_model.with_structured_output(
    SymptomDetails
)


def extract_symptom_details(
    user_message: str,
    chat_history: list[BaseMessage],
) -> SymptomDetails:
    """Extract facts for the current medical problem only."""

    extraction_messages = [
        SystemMessage(
            content=(
                "Extract medical information for the CURRENT medical "
                "problem only. Use recent conversation history only when "
                "it belongs to the same medical problem.\n\n"

                "Do not carry information from an older unrelated "
                "complaint into the current complaint.\n\n"

                "For boolean fields:\n"
                "- true = explicitly present.\n"
                "- false = explicitly denied.\n"
                "- null = not mentioned.\n\n"

                "Examples:\n"
                "\"I have a cough\" -> cough=true.\n"
                "\"I do not have a cough\" -> cough=false.\n"
                "\"I have a fever\" -> cough=null.\n"
                "\"My shoulder hurts\" -> numbness=null.\n"
                "\"My shoulder hurts but I have no numbness\" "
                "-> numbness=false.\n\n"

                "Never guess missing information. "
                "Never infer that an unmentioned symptom is absent. "
                "Do not diagnose or infer diseases."
            )
        ),
        *chat_history,
        HumanMessage(
            content=user_message
        ),
    ]

    return symptom_extractor.invoke(
        extraction_messages
    )



# New medical problem detection

class ConversationState(BaseModel):
    """Whether the latest message starts a different medical problem."""

    new_medical_problem: bool = Field(
        description=(
            "True only when the latest message starts a clearly "
            "different medical problem from the active complaint."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation for the decision."
        )
    )


conversation_detector = chat_model.with_structured_output(
    ConversationState
)


def detect_new_medical_problem(
    user_message: str,
    chat_history: list[BaseMessage],
) -> ConversationState:
    """Detect whether the user switched to an unrelated complaint."""

    if not chat_history:
        return ConversationState(
            new_medical_problem=False,
            reason=(
                "There is no previous medical problem."
            ),
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
                "Fever -> body aches = SAME.\n"
                "Fever -> shoulder pain after lifting boxes = NEW.\n"
                "Back pain -> numbness in the leg = SAME.\n"
                "Back pain -> pain is worse when moving = SAME.\n"
                "Heavy menstrual bleeding -> dizziness = SAME.\n"
                "Heavy menstrual bleeding -> ankle pain after fall = NEW.\n"
                "Pregnancy -> vaginal bleeding = SAME.\n"
                "Pregnancy -> injured wrist after a fall = NEW.\n\n"

                "Do not start a new problem merely because a related "
                "symptom appears."
            )
        ),
        *chat_history,
        HumanMessage(
            content=user_message
        ),
    ]

    return conversation_detector.invoke(
        detector_messages
    )



# Risk assessment


class RiskAssessment(BaseModel):
    """Basic triage-style risk assessment."""

    risk_level: str = Field(
        description=(
            "One of: low, moderate, high."
        )
    )

    urgent: bool = Field(
        description=(
            "True only when explicitly reported symptoms may require "
            "urgent medical assessment."
        )
    )

    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit warning signs that contributed to the assessment."
        ),
    )

    emergency_reason: Optional[str] = Field(
        default=None,
        description=(
            "Short description of why the situation may be urgent."
        ),
    )


risk_detector = chat_model.with_structured_output(
    RiskAssessment
)


def assess_medical_risk(
    user_message: str,
    symptom_details: SymptomDetails,
) -> RiskAssessment:
    """
    Assess whether explicitly reported symptoms contain urgent warning signs.
    """

    symptom_json = json.dumps(
        symptom_details.model_dump(
            exclude_none=True
        ),
        ensure_ascii=False,
    )

    messages = [
        SystemMessage(
            content=(
                "Perform cautious medical triage using ONLY symptoms "
                "explicitly provided by the user.\n\n"

                "Return HIGH risk and urgent=true for clear emergency "
                "warning signs such as:\n"
                "- chest pain with difficulty breathing\n"
                "- severe difficulty breathing\n"
                "- new one-sided weakness or trouble speaking\n"
                "- loss of consciousness\n"
                "- seizure\n"
                "- severe uncontrolled bleeding\n"
                "- fever with confusion or stiff neck\n"
                "- pregnancy with severe bleeding or severe abdominal pain\n"
                "- sudden severe testicular pain\n"
                "- prolonged painful erection\n\n"

                "Return MODERATE when prompt medical review may be "
                "appropriate but there is no clear emergency warning sign.\n\n"

                "Return LOW for common mild symptoms without clear "
                "warning signs.\n\n"

                "Do not invent symptoms. "
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

    return risk_detector.invoke(
        messages
    )


def generate_urgent_response(
    user_message: str,
    risk_assessment: RiskAssessment,
) -> str:
    """Generate short urgent guidance in the user's language."""

    risk_json = json.dumps(
        risk_assessment.model_dump(
            exclude_none=True
        ),
        ensure_ascii=False,
    )

    response = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a cautious medical information assistant.\n\n"

                    "A separate triage system has already determined that "
                    "the user's reported symptoms may require urgent "
                    "medical assessment.\n\n"

                    "Respond in the SAME LANGUAGE as the user's latest "
                    "message.\n\n"

                    "Your response must:\n"
                    "- briefly acknowledge the symptoms\n"
                    "- clearly advise urgent medical assessment now\n"
                    "- advise contacting emergency medical services if "
                    "symptoms are severe, rapidly worsening, or the person "
                    "feels faint or critically unwell\n"
                    "- not diagnose a disease\n"
                    "- not list speculative possible diagnoses\n"
                    "- not delay care by asking additional questions\n"
                    "- be concise and under 120 words\n"
                )
            ),
            HumanMessage(
                content=(
                    f"User message:\n{user_message}\n\n"
                    f"Risk assessment:\n{risk_json}"
                )
            ),
        ]
    )

    return str(
        response.content
    ).strip()



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


def detect_topic(
    query: str,
) -> Optional[str]:
    """Detect a broad medical topic from an English query."""

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

# Missing-information logic

def get_missing_information(
    details: SymptomDetails,
) -> list[str]:
    """Determine useful missing information for common symptom groups."""

    missing = []

    symptom = (
        details.symptom or ""
    ).lower()

    location = (
        details.body_location or ""
    ).lower()

    # Fever
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

    # Musculoskeletal pain
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

    # Genital pain
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


# English retrieval query


def translate_query_for_retrieval(
    query: str,
) -> str:
    """Create a concise English medical search query."""

    response = chat_model.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the user's latest message as a concise "
                    "English medical search query for retrieval from a "
                    "medical database. Translate into English when needed. "
                    "Keep medically important symptoms, location, duration, "
                    "temperature, severity, triggers, injury and warning signs. "
                    "Remove conversational filler. "
                    "Do not answer the question. "
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

    return english_query or query



def retrieve_and_rerank(
    query: str,
):
    """Retrieve candidate documents and rerank them."""

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

    normalized_query = (
        english_query.lower()
    )

    query_words = {
        word
        for word in normalized_query
        .replace("?", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .split()
        if len(word) >= 4
    }

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

    reranked_results = []

    for document, vector_score in results:
        metadata = (
            document.metadata
            or {}
        )

        title = str(
            metadata.get(
                "title",
                "",
            )
        ).lower()

        searchable_metadata = " ".join(
            [
                str(
                    metadata.get(
                        "title",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "category",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "categories",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "mapped_title",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "document_type",
                        "",
                    )
                ),
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
            if metadata.get(
                "document_type"
            ) == "curated_topic"
            else 0.0
        )

        exact_topic_bonus = 0.0
        mismatch_penalty = 0.0

        # Fever-specific ranking.
        if detected_topic == "fever":
            if title == "fever":
                exact_topic_bonus += 0.45

            elif title == "high fever":
                exact_topic_bonus += 0.20

            rare_fever_terms = {
                "hemorrhagic fever",
                "hemorrhagic fevers",
                "valley fever",
                "yellow fever",
                "dengue",
                "malaria",
            }

            if any(
                rare_term in title
                and rare_term not in normalized_query
                for rare_term in rare_fever_terms
            ):
                mismatch_penalty += 0.40

            if (
                "heat stroke" in title
                and not query_mentions_heat
            ):
                mismatch_penalty += 0.35

            if (
                "heat exhaustion" in title
                and not query_mentions_heat
            ):
                mismatch_penalty += 0.35

        # Avoid fractures/dislocations when trauma is not reported.
        if not query_mentions_trauma:
            if any(
                term in searchable_metadata
                for term in serious_injury_terms
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

    selected_documents = []
    seen_sources = set()

    for (
        document,
        final_score,
    ) in reranked_results:
        metadata = (
            document.metadata
            or {}
        )

        unique_key = (
            metadata.get(
                "source_url"
            )
            or metadata.get(
                "title"
            )
            or document.page_content[:100]
        )

        if unique_key in seen_sources:
            continue

        if final_score < 0.50:
            continue

        seen_sources.add(
            unique_key
        )

        selected_documents.append(
            document
        )

        if len(
            selected_documents
        ) == 5:
            break

    app.logger.info(
        "Selected sources: %s",
        [
            document.metadata.get(
                "title"
            )
            for document
            in selected_documents
        ],
    )

    return selected_documents


retriever = RunnableLambda(
    retrieve_and_rerank
)

# RAG chain

contextualize_question_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                contextualize_q_system_prompt,
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
)


history_aware_retriever = (
    create_history_aware_retriever(
        chat_model,
        retriever,
        contextualize_question_prompt,
    )
)


answer_prompt = (
    ChatPromptTemplate.from_messages(
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

                    "Rules:\n"
                    "- Do not ask for information already present in "
                    "symptom_details.\n"
                    "- If clarification is needed, ask at most two of "
                    "the most important items from missing_information.\n"
                    "- Do not mix symptoms from an older unrelated "
                    "problem into the current problem.\n"
                    "- If enough information is available, give "
                    "practical guidance instead of asking unnecessary "
                    "questions."
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
)


question_answer_chain = (
    create_stuff_documents_chain(
        chat_model,
        answer_prompt,
    )
)


rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

# Chat history


def convert_chat_history(
    history_data,
) -> list[BaseMessage]:
    """Convert browser history into LangChain messages."""

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


# Casual conversation in multiple languages.


CASUAL_MESSAGES = {
    # English
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

    # Swedish
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

    # Spanish
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

    # Arabic
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

    "dolor",
    "fiebre",
    "sangrado",
    "tos",
    "sexo",

    "ألم",
    "حرارة",
    "حمى",
    "نزيف",
    "سعال",
    "جنس",
}


def normalize_message(
    message: str,
) -> str:
    """Normalize text for message matching."""

    return " ".join(
        message.lower()
        .strip(" !?.,")
        .split()
    )


def is_casual_message(
    message: str,
) -> bool:
    """Return True only for non-medical conversational messages."""

    normalized = normalize_message(
        message
    )

    contains_medical_information = any(
        medical_hint in normalized
        for medical_hint in MEDICAL_HINTS
    )

    if contains_medical_information:
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
    """Return a short casual response in the user's language."""

    normalized = normalize_message(
        message
    )

    arabic_characters = (
        "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
    )

    if any(
        character in message
        for character in arabic_characters
    ):
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

    spanish_terms = {
        "hola",
        "gracias",
        "adiós",
        "adios",
        "pregunta",
        "ayuda",
    }

    if any(
        term in normalized
        for term in spanish_terms
    ):
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

    swedish_terms = {
        "hej",
        "hallå",
        "tack",
        "fråga",
        "hjälp",
        "god morgon",
        "god kväll",
    }

    if any(
        term in normalized
        for term in swedish_terms
    ):
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
    """Check whether history contains a medical user message."""

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
                "answer": (
                    get_casual_response(
                        user_message
                    )
                ),
                "sources": [],
                "reset_history": False,
                "urgent": False,
            }
        )

    try:
        full_chat_history = (
            convert_chat_history(
                history_data
            )
        )

        
        # Detect new medical complaint 
        

        if history_has_medical_user_message(
            full_chat_history
        ):
            conversation_state = (
                detect_new_medical_problem(
                    user_message=user_message,
                    chat_history=full_chat_history,
                )
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

        # New complaint = do not use old case history.
        if (
            conversation_state
            .new_medical_problem
        ):
            active_chat_history = []
        else:
            active_chat_history = (
                full_chat_history
            )

        # Extract symptoms
      

        symptom_details = (
            extract_symptom_details(
                user_message=user_message,
                chat_history=active_chat_history,
            )
        )

        symptom_data = (
            symptom_details.model_dump(
                exclude_none=True
            )
        )

        app.logger.info(
            "Extracted symptoms: %s",
            symptom_data,
        )

        

        risk_assessment = (
            assess_medical_risk(
                user_message=user_message,
                symptom_details=symptom_details,
            )
        )

        app.logger.info(
            "Risk assessment: %s",
            risk_assessment.model_dump(),
        )

        
        # Urgent cases bypass normal RAG questioning

        if risk_assessment.urgent:
            urgent_answer = (
                generate_urgent_response(
                    user_message=user_message,
                    risk_assessment=risk_assessment,
                )
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
        

        missing_information = (
            get_missing_information(
                symptom_details
            )
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

        app.logger.info(
            "Missing information: %s",
            missing_information,
        )

        response = rag_chain.invoke(
            {
                "input": user_message,
                "chat_history": active_chat_history,
                "symptom_details": symptom_context,
                "missing_information": missing_context,
            }
        )

        answer = str(
            response.get(
                "answer",
                "",
            )
        ).strip()

        retrieved_documents = (
            response.get(
                "context",
                [],
            )
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

        
        # Sources
        

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