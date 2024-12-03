from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ConfigDict, Field
from datetime import datetime
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import numpy as np
import google.generativeai as genai
from sqlalchemy import create_engine, Column, String, DateTime, JSON, UUID, BigInteger, Text, ARRAY, Boolean, Table, MetaData, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, selectinload
import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import ValidationError
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict
import spacy
from uuid import UUID as PyUUID  # Import Python's UUID type
import ssl
from sqlalchemy.sql import text
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from collections import Counter
import traceback
import logging
from mangum import Mangum

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Mood Journal API",
    description="API for mood journaling and analysis",
    version="1.0.0",
    root_path="/api" if os.environ.get("VERCEL_ENV") else ""
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable must be set")

genai.configure(api_key=GEMINI_API_KEY)

# Gemini model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize sentiment analysis models
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
bart_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

roberta_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
roberta_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Create Base class for SQLAlchemy models
Base = declarative_base()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set")

# Create SQLAlchemy engine with proper SSL settings
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={
        "sslmode": "require"
    }
)

# Create tables
Base.metadata.create_all(bind=engine)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Add this after SessionLocal definition and before the routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Valid day rating options
DAY_RATING_OPTIONS = ['great', 'good', 'okay', 'notGreat', 'terrible']

# Valid mood options
VALID_MOODS = ['Happy', 'Sad', 'Angry', 'Anxious', 'Excited', 'Tired', 'Calm', 'Stressed']

# Define emotions mapping for RoBERTa
emotions = {
    0: "admiration", 1: "amusement", 2: "anger",
    3: "annoyance", 4: "approval", 5: "caring",
    6: "confusion", 7: "curiosity", 8: "desire",
    9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear",
    15: "gratitude", 16: "grief", 17: "joy",
    18: "love", 19: "nervousness", 20: "optimism",
    21: "pride", 22: "realization", 23: "relief",
    24: "remorse", 25: "sadness", 26: "surprise",
    27: "neutral"
}

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID, unique=True, nullable=False, server_default=text('gen_random_uuid()'))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text('current_timestamp'))
    last_login = Column(DateTime, nullable=True)

class MoodTracker(Base):
    __tablename__ = "mood_tracker"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    email = Column(String, nullable=False)
    username = Column(Text, nullable=False)
    journal_id = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    user_journal = Column(Text, nullable=False, server_default='')
    day_rating = Column(String(50), nullable=False)
    feelings = Column(Text, nullable=False, server_default='')
    selected_moods = Column(ARRAY(Text), nullable=False)
    created_at = Column(DateTime, nullable=True, server_default=text('current_timestamp'))
    updated_at = Column(DateTime, nullable=True, server_default=text('current_timestamp'))
    sentiment = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    roberta_emotions = Column(JSON, nullable=True)
    selected_moods_analysis = Column(JSON, nullable=True)
    detailed_analysis = Column(JSON, nullable=True)
    emotion_transitions = Column(JSON, nullable=True)
    emotion_cooccurrence = Column(JSON, nullable=True)
    emotional_stability = Column(JSON, nullable=True)
    topic_analysis = Column(JSON, nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    archived = Column(Boolean, nullable=True, server_default='false')
    last_sentiment_updated_at = Column(DateTime, nullable=True, server_default=text('current_timestamp'))
    bart_analysis = Column(JSON, nullable=True)
    sentiment_timeline = Column(JSON, nullable=True)
    emotion_intensity_metrics = Column(JSON, nullable=True)
    linguistic_features = Column(JSON, nullable=True)
    behavioral_patterns = Column(JSON, nullable=True)
    contextual_triggers = Column(JSON, nullable=True)
    social_references = Column(JSON, nullable=True)
    custom_tags = Column(JSON, nullable=True)
    graph_data = Column(JSON, nullable=True)

# Base configuration class for all models
class BaseModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class EmotionTransition(BaseModelConfig):
    from_emotion: str
    to_emotion: str
    transition_score: float

class EmotionalStability(BaseModelConfig):
    variance: float
    stability_score: float
    emotional_shifts: int

class DetailedAnalysis(BaseModelConfig):
    sentence_level: List[Dict[str, Any]]
    topic_analysis: Dict[str, Any]
    emotion_transitions: List[EmotionTransition]
    emotion_cooccurrence: Dict[str, Dict[str, int]]
    emotional_stability: EmotionalStability

class AnalysisRequest(BaseModelConfig):
    user_id: str
    email: str
    username: str
    content: str
    day_rating: str
    selected_moods: List[str]
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    tags: Optional[List[str]] = Field(default_factory=list)
    journal_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "username": "user123",
                "content": "I am feeling good today",
                "day_rating": "good",
                "selected_moods": ["Happy"],
                "tags": ["daily"],
                "timestamp": "2024-03-14T12:00:00Z"
            }
        }

    @field_validator('user_id')
    @classmethod
    def validate_uuid(cls, v):
        try:
            return str(uuid.UUID(v))
        except ValueError:
            raise ValueError('Invalid UUID format')

    @field_validator('day_rating')
    @classmethod
    def validate_day_rating(cls, v):
        valid_ratings = ['good', 'bad', 'neutral']  # Add your valid ratings
        if v.lower() not in valid_ratings:
            raise ValueError(f'Invalid day rating. Must be one of: {valid_ratings}')
        return v.lower()

    @field_validator('selected_moods')
    @classmethod
    def validate_moods(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError('Selected moods must be a non-empty list')
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if v is None:
            return v
        if not v or '@' not in v:
            raise ValueError('Invalid email address')
        return v

class AnalysisResponse(BaseModelConfig):
    id: int
    user_id: PyUUID
    email: str
    username: str
    journal_id: PyUUID
    user_journal: str
    day_rating: str
    feelings: str
    selected_moods: List[str]
    created_at: datetime
    updated_at: Optional[datetime]
    sentiment: Optional[Dict[str, Any]]
    recommendations: Optional[Dict[str, Any]]
    roberta_emotions: Optional[Dict[str, Any]]
    selected_moods_analysis: Optional[Dict[str, Any]]
    detailed_analysis: Optional[Dict[str, Any]]
    emotion_transitions: Optional[Dict[str, Any]]
    emotion_cooccurrence: Optional[Dict[str, Any]]
    emotional_stability: Optional[Dict[str, Any]]
    topic_analysis: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    archived: Optional[bool]
    last_sentiment_updated_at: Optional[datetime]
    bart_analysis: Optional[Dict[str, Any]]
    sentiment_timeline: Optional[Dict[str, Any]]
    emotion_intensity_metrics: Optional[Dict[str, Any]]
    linguistic_features: Optional[Dict[str, Any]]
    behavioral_patterns: Optional[Dict[str, Any]]
    contextual_triggers: Optional[Dict[str, Any]]
    social_references: Optional[Dict[str, Any]]
    custom_tags: Optional[Dict[str, Any]]
    graph_data: Optional[Dict[str, Any]]

# Add these Pydantic models near the top of the file with your other BaseModel definitions
class UserCreate(BaseModelConfig):
    email: str

class UserResponse(BaseModelConfig):
    id: int
    user_id: PyUUID
    email: str
    username: str
    created_at: datetime
    last_login: Optional[datetime]

# Add this new Pydantic model for user creation requests
class UserCreateRequest(BaseModelConfig):
    userData: Dict[str, Any] = Field(...)

    class Config:
        json_schema_extra = {
            "example": {
                "userData": {
                    "email": "user@example.com",
                    "name": "John Doe"
                }
            }
        }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def get_gemini_recommendations(analysis_result: dict, day_rating: str, selected_moods: list):
    try:
        chat_session = gemini_model.start_chat(history=[])
        
        # Extract key metrics with safe defaults
        sentiment_dist = analysis_result.get("sentiment", {}).get("scores", {})
        emotion_scores = analysis_result.get("roberta_emotions", {}).get("top_5", {})
        emotional_stability = analysis_result.get("emotional_stability", {})
        
        prompt = f"""You are an AI wellness coach. Based on the emotional data below, provide a detailed analysis and specific recommendations.
Please follow the EXACT format specified, ensuring all sections are complete.

EMOTIONAL DATA:
- Day Rating: {day_rating}
- Selected Moods: {', '.join(selected_moods)}
- Primary Emotions: {', '.join([f'{k}: {v:.2f}' for k, v in emotion_scores.items()])}
- Emotional Stability: {emotional_stability.get('stability_score', 0):.2f}
- Sentiment: Positive {sentiment_dist.get('positive', 0):.2f} / Negative {sentiment_dist.get('negative', 0):.2f}

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

EMOTIONAL INSIGHT:
[Write 2-3 sentences analyzing the emotional patterns, their implications, and potential impact]

ACTIVITIES:
1. [Specific Activity Name] - [Exact Duration] - [Intensity Level] - [Scientific Benefit]
2. [Specific Activity Name] - [Exact Duration] - [Intensity Level] - [Scientific Benefit]
3. [Specific Activity Name] - [Exact Duration] - [Intensity Level] - [Scientific Benefit]

BOOKS/MEDIA:
1. [Specific Title] - [Media Type] - [Evidence-based Reason for Recommendation]
2. [Specific Title] - [Media Type] - [Evidence-based Reason for Recommendation]


MOOD-BOOSTING FOODS:
1. [Specific Food] - [Exact Calories] - [Key Nutrients] - [Research-backed Benefits]
2. [Specific Food] - [Exact Calories] - [Key Nutrients] - [Research-backed Benefits]
3. [Specific Food] - [Exact Calories] - [Key Nutrients] - [Research-backed Benefits]

SELF-CARE PRACTICES:
1. [Specific Practice] - [Exact Duration] - [Clear Psychological Benefit]
2. [Specific Practice] - [Exact Duration] - [Clear Psychological Benefit]

Remember:
1. Be specific and detailed in each recommendation
2. Include ALL sections
3. Follow the exact format with dashes between elements
4. Base recommendations on scientific research
5. Ensure recommendations align with the emotional data"""

        response = chat_session.send_message(prompt)
        
        if not response or not response.text:
            raise ValueError("Empty response from Gemini")

        # Initialize structured response
        parsed_response = {
            'emotional_insight': '',
            'suggestions': {
                'activities': [],
                'books_media': [],
                'foods': [],
                'self_care': []
            }
        }

        # Split response into sections
        sections = response.text.split('\n\n')
        current_section = None

        for section in sections:
            section = section.strip()
            
            # Parse Emotional Insight
            if 'EMOTIONAL INSIGHT:' in section:
                parsed_response['emotional_insight'] = section.replace('EMOTIONAL INSIGHT:', '').strip()
                continue

            # Parse Activities
            if 'ACTIVITIES:' in section:
                current_section = 'activities'
                items = [line.strip() for line in section.split('\n')[1:] if line.strip() and line[0].isdigit()]
                for item in items:
                    parts = item.split(' - ', 3)  # Split into 4 parts max
                    if len(parts) >= 4:
                        parsed_response['suggestions']['activities'].append({
                            'activity': parts[0].lstrip('123456789. '),
                            'duration': parts[1],
                            'intensity': parts[2],
                            'benefit': parts[3]
                        })

            # Parse Books/Media
            elif 'BOOKS/MEDIA:' in section:
                current_section = 'books_media'
                items = [line.strip() for line in section.split('\n')[1:] if line.strip() and line[0].isdigit()]
                for item in items:
                    parts = item.split(' - ', 2)  # Split into 3 parts
                    if len(parts) >= 3:
                        parsed_response['suggestions']['books_media'].append({
                            'title': parts[0].lstrip('123456789. '),
                            'type': parts[1],
                            'reason': parts[2]
                        })

            # Parse Foods
            elif 'MOOD-BOOSTING FOODS:' in section:
                current_section = 'foods'
                items = [line.strip() for line in section.split('\n')[1:] if line.strip() and line[0].isdigit()]
                for item in items:
                    parts = item.split(' - ', 3)  # Split into 4 parts max
                    if len(parts) >= 4:
                        parsed_response['suggestions']['foods'].append({
                            'food': parts[0].lstrip('123456789. '),
                            'calories': parts[1],
                            'nutrients': parts[2],
                            'benefits': parts[3]
                        })

            # Parse Self-Care Practices
            elif 'SELF-CARE PRACTICES:' in section:
                current_section = 'self_care'
                items = [line.strip() for line in section.split('\n')[1:] if line.strip() and line[0].isdigit()]
                for item in items:
                    parts = item.split(' - ', 2)  # Split into 3 parts
                    if len(parts) >= 3:
                        parsed_response['suggestions']['self_care'].append({
                            'practice': parts[0].lstrip('123456789. '),
                            'duration': parts[1],
                            'benefit': parts[2]
                        })

        # Validate response completeness
        if (not parsed_response['emotional_insight'] or
            len(parsed_response['suggestions']['activities']) < 3 or
            len(parsed_response['suggestions']['books_media']) < 2 or
            len(parsed_response['suggestions']['foods']) < 3 or
            len(parsed_response['suggestions']['self_care']) < 2):
            
            logging.warning("Incomplete Gemini response, retrying...")
            # Retry once with the same prompt
            response = chat_session.send_message(prompt)
            if not response or not response.text:
                return get_fallback_recommendations(day_rating, selected_moods)
            
            # If still incomplete, use fallback
            if not all(parsed_response['suggestions'].values()):
                return get_fallback_recommendations(day_rating, selected_moods)

        return parsed_response

    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return get_fallback_recommendations(day_rating, selected_moods)

def get_fallback_recommendations(day_rating: str, moods: list) -> dict:
    """Provide fallback recommendations based on day rating and moods."""
    # Customize fallback responses based on mood
    mood_based_activities = {
        "stressed": {
            "activity": "Deep breathing exercises",
            "duration": "10 minutes",
            "intensity": "Low",
            "benefit": "Activates parasympathetic nervous system"
        },
        "anxious": {
            "activity": "Progressive muscle relaxation",
            "duration": "15 minutes",
            "intensity": "Low",
            "benefit": "Reduces physical tension and anxiety"
        },
        # Add more mood-specific activities...
    }

    # Select appropriate activity based on moods
    primary_mood = moods[0].lower() if moods else "neutral"
    activity = mood_based_activities.get(primary_mood, {
        "activity": "Mindful walking",
        "duration": "20 minutes",
        "intensity": "Moderate",
        "benefit": "Combines exercise with mindfulness"
    })

    return {
        "emotional_insight": f"Based on your {day_rating} day and {primary_mood} mood, focusing on gentle self-care may be beneficial.",
        "suggestions": {
            "activities": [activity],
            "books_media": [{
                "title": "Headspace App",
                "type": "Meditation App",
                "reason": "Structured mindfulness practices"
            }],
            "foods": [{
                "food": "Dark chocolate",
                "calories": "150 per 30g",
                "nutrients": "Magnesium, Antioxidants",
                "properties": "Boosts mood via endorphin release"
            }],
            "self_care": [{
                "practice": "Mindful breathing",
                "duration": "5 minutes",
                "benefit": "Quick stress reduction"
            }]
        }
    }

# Disable SSL verification for NLTK downloads (if needed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data with proper error handling
def download_nltk_data():
    required_packages = [
        'punkt',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

# Call the download function
download_nltk_data()

# Initialize NLTK components after downloads
try:
    # Initialize tokenizer and other NLTK components
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Error initializing NLTK components: {str(e)}")
    raise

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def calculate_emotion_cooccurrence(sentence_analysis):
    """
    Calculate emotion co-occurrence patterns with enhanced metrics and validation.
    
    Args:
        sentence_analysis (List[Dict]): List of sentence-level emotion analysis
        
    Returns:
        Dict containing co-occurrence matrix and additional metrics
    """
    try:
        # Initialize data structures
        cooccurrence = defaultdict(lambda: defaultdict(int))
        emotion_strengths = defaultdict(list)
        temporal_patterns = defaultdict(list)
        
        # Track sentence position for temporal analysis
        total_sentences = len(sentence_analysis)
        
        for idx, sent in enumerate(sentence_analysis):
            # Get emotion scores with validation
            emotion_scores = sent.get("emotion_scores", {})
            if not isinstance(emotion_scores, dict):
                continue
                
            # Filter significant emotions (score > 0.1)
            significant_emotions = {
                emotion: score 
                for emotion, score in emotion_scores.items() 
                if score > 0.1
            }
            
            # Record temporal position (0-1 range)
            temporal_position = idx / max(1, total_sentences - 1)
            
            # Calculate co-occurrences and collect metrics
            emotions = list(significant_emotions.keys())
            for i in range(len(emotions)):
                emotion1 = emotions[i]
                score1 = significant_emotions[emotion1]
                
                # Record emotion strength
                emotion_strengths[emotion1].append(score1)
                
                # Record temporal pattern
                temporal_patterns[emotion1].append({
                    "position": temporal_position,
                    "strength": score1
                })
                
                # Calculate weighted co-occurrences
                for j in range(i + 1, len(emotions)):
                    emotion2 = emotions[j]
                    score2 = significant_emotions[emotion2]
                    
                    # Weight co-occurrence by average intensity
                    weight = (score1 + score2) / 2
                    cooccurrence[emotion1][emotion2] += weight
                    cooccurrence[emotion2][emotion1] += weight

        # Calculate additional metrics
        metrics = {
            "average_intensities": {
                emotion: sum(scores) / len(scores) if scores else 0
                for emotion, scores in emotion_strengths.items()
            },
            "temporal_patterns": {
                emotion: {
                    "early": sum(1 for p in patterns if p["position"] < 0.33),
                    "middle": sum(1 for p in patterns if 0.33 <= p["position"] < 0.66),
                    "late": sum(1 for p in patterns if p["position"] >= 0.66)
                }
                for emotion, patterns in temporal_patterns.items()
            }
        }
        
        # Normalize co-occurrence values
        max_cooccurrence = max(
            (v for d in cooccurrence.values() for v in d.values()),
            default=1
        )
        
        normalized_cooccurrence = {
            emotion1: {
                emotion2: round(count / max_cooccurrence, 3)
                for emotion2, count in emotion_pairs.items()
            }
            for emotion1, emotion_pairs in cooccurrence.items()
            if any(emotion_pairs.values())  # Only include emotions with co-occurrences
        }

        return {
            "cooccurrence_matrix": normalized_cooccurrence,
            "metrics": metrics,
            "emotion_pairs": [
                {
                    "emotions": [emotion1, emotion2],
                    "strength": strength,
                    "frequency": len([
                        s for s in sentence_analysis
                        if emotion1 in s.get("emotion_scores", {}) 
                        and emotion2 in s.get("emotion_scores", {})
                    ]) / total_sentences
                }
                for emotion1, pairs in normalized_cooccurrence.items()
                for emotion2, strength in pairs.items()
                if emotion1 < emotion2  # Avoid duplicates
            ]
        }

    except Exception as e:
        logging.error(f"Error in calculate_emotion_cooccurrence: {str(e)}")
        return {
            "cooccurrence_matrix": {},
            "metrics": {
                "average_intensities": {},
                "temporal_patterns": {}
            },
            "emotion_pairs": []
        }

def calculate_emotional_stability(sentence_analysis):
    sentiment_scores = [s["vader_sentiment"]["compound"] for s in sentence_analysis]
    if not sentiment_scores:  # Handle empty analysis
        return {
            "variance": 0.0,
            "stability_score": 1.0,
            "emotional_shifts": 0
        }
    
    return {
        "variance": float(np.var(sentiment_scores)),
        "stability_score": 1.0 / (1.0 + float(np.var(sentiment_scores))),  # Normalized 0-1
        "emotional_shifts": len([
            i for i in range(len(sentiment_scores)-1)
            if abs(sentiment_scores[i] - sentiment_scores[i+1]) > 0.5  # Significant shift threshold
        ])
    }

# Add these functions near the top of the file, after the imports
def analyze_emotions_with_roberta(text: str):
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = roberta_model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    scores = softmax(scores)
    
    # Map scores to emotions
    emotion_scores = {emotions[i]: float(score) for i, score in enumerate(scores)}
    return emotion_scores

def analyze_text_features(doc, sentence_analysis):
    try:
        # Topic Analysis
        topic_analysis = {
            "key_topics": [chunk.text for chunk in doc.noun_chunks],
            "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
            "main_themes": [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ in ['nsubj', 'dobj']]
        }

        # Linguistic Features
        linguistic_features = {
            "sentence_structure": [{
                "text": sent.text,
                "word_count": len([token for token in sent if not token.is_punct]),
                "complexity": len([token for token in sent if token.is_stop]) / len(sent) if len(sent) > 0 else 0
            } for sent in doc.sents],
            "vocabulary_richness": len(set([token.text.lower() for token in doc if not token.is_stop])) / len(doc) if len(doc) > 0 else 0,
            "key_phrases": [chunk.text for chunk in doc.noun_chunks]
        }

        # Behavioral Patterns
        behavioral_patterns = {
            "identified_patterns": {
                "active_behaviors": [
                    chunk.text for chunk in doc.noun_chunks
                    if any(verb in chunk.root.head.text.lower() 
                          for verb in ["do", "make", "create", "achieve"])
                ],
                "passive_behaviors": [
                    chunk.text for chunk in doc.noun_chunks
                    if any(verb in chunk.root.head.text.lower() 
                          for verb in ["feel", "think", "wonder", "wait"])
                ],
                "social_behaviors": [
                    chunk.text for chunk in doc.noun_chunks
                    if any(verb in chunk.root.head.text.lower() 
                          for verb in ["talk", "meet", "share", "help"])
                ]
            }
        }

        # Contextual Triggers
        contextual_triggers = {
            "time_references": [ent.text for ent in doc.ents if ent.label_ == 'TIME' or ent.label_ == 'DATE'],
            "location_mentions": [ent.text for ent in doc.ents if ent.label_ == 'GPE' or ent.label_ == 'LOC'],
            "situational_context": [
                token.text for token in doc 
                if token.dep_ in ['prep', 'advmod'] and not token.is_stop
            ]
        }

        # Social References
        social_references = {
            "mentioned_entities": {
                "people": [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
                "organizations": [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
                "relationships": [
                    token.text for token in doc 
                    if token.text.lower() in ["friend", "family", "colleague", "partner"]
                ]
            }
        }

        # Emotion Intensity Metrics
        emotion_intensity_metrics = {
            "average_intensity": np.mean([
                max(sent["emotion_scores"].values()) 
                for sent in sentence_analysis
            ]),
            "peak_emotions": [
                {
                    "emotion": max(sent["emotion_scores"].items(), key=lambda x: x[1])[0],
                    "intensity": max(sent["emotion_scores"].values()),
                    "text": sent["text"]
                }
                for sent in sentence_analysis
            ],
            "emotional_range": max(
                max(sent["emotion_scores"].values()) for sent in sentence_analysis
            ) - min(
                min(sent["emotion_scores"].values()) for sent in sentence_analysis
            )
        }

        # Generate tags based on analysis
        tags = (
            topic_analysis["key_topics"][:5] +  # Top 5 topics
            [ent["text"] for ent in topic_analysis["entities"][:3]] +  # Top 3 entities
            [emotion["emotion"] for emotion in emotion_intensity_metrics["peak_emotions"][:3]]  # Top 3 emotions
        )

        return {
            "topic_analysis": topic_analysis,
            "linguistic_features": linguistic_features,
            "behavioral_patterns": behavioral_patterns,
            "contextual_triggers": contextual_triggers,
            "social_references": social_references,
            "emotion_intensity_metrics": emotion_intensity_metrics,
            "tags": list(set(tags))  # Remove duplicates
        }

    except Exception as e:
        print(f"Error in analyze_text_features: {str(e)}")
        return {
            "topic_analysis": {},
            "linguistic_features": {},
            "behavioral_patterns": {},
            "contextual_triggers": {},
            "social_references": {},
            "emotion_intensity_metrics": {},
            "tags": []
        }

# Update the analyze_sentiment function to include these features
async def analyze_sentiment(text: str, day_rating: str, selected_moods: List[str]):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        
        # Get emotion scores for the full text
        emotion_scores = analyze_emotions_with_roberta(text)
        
        # Basic sentiment analysis with VADER
        sentence_analysis = [{
            "text": sent.text,
            "vader_sentiment": sia.polarity_scores(sent.text),
            "emotion_scores": analyze_emotions_with_roberta(sent.text)
        } for sent in sentences]

        # Calculate sentiment scores
        sentiment = {
            "scores": {
                "positive": float(np.mean([s["vader_sentiment"]["pos"] for s in sentence_analysis])),
                "neutral": float(np.mean([s["vader_sentiment"]["neu"] for s in sentence_analysis])),
                "negative": float(np.mean([s["vader_sentiment"]["neg"] for s in sentence_analysis]))
            }
        }

        # Get text features analysis
        text_features = analyze_text_features(doc, sentence_analysis)
        
        # Calculate emotion transitions and stability
        emotion_transitions = []
        for i in range(len(sentences) - 1):
            current_emotions = sentence_analysis[i]["emotion_scores"]
            next_emotions = sentence_analysis[i + 1]["emotion_scores"]
            
            current_dominant = max(current_emotions.items(), key=lambda x: x[1])[0]
            next_dominant = max(next_emotions.items(), key=lambda x: x[1])[0]
            
            emotion_transitions.append({
                "from_emotion": current_dominant,
                "to_emotion": next_dominant,
                "transition_score": float(abs(
                    max(current_emotions.values()) - max(next_emotions.values())
                ))
            })

        # Calculate emotional stability
        emotional_stability = calculate_emotional_stability(sentence_analysis)

        # Get emotion cooccurrence
        emotion_cooccurrence = calculate_emotion_cooccurrence(sentence_analysis)

        # Get Gemini recommendations based on analysis
        recommendations = await get_gemini_recommendations({
            "sentiment": sentiment,
            "roberta_emotions": {
                "top_5": dict(sorted(
                    emotion_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            },
            "emotional_stability": emotional_stability
        }, day_rating, selected_moods)

        # Combine all analysis results
        analysis_result = {
            "sentiment": sentiment,
            "roberta_emotions": {
                "top_5": dict(sorted(
                    emotion_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]),
                "all_emotions": emotion_scores
            },
            "selected_moods_analysis": {
                "analysis": f"Analysis of moods: {', '.join(selected_moods)}",
                "selected_moods": selected_moods,
                "detected_emotions": emotion_scores
            },
            "recommendations": recommendations,
            "detailed_analysis": {
                "sentence_level": sentence_analysis,
                "topic_analysis": text_features["topic_analysis"],
                "emotion_transitions": emotion_transitions,
                "emotion_cooccurrence": emotion_cooccurrence,
                "emotional_stability": emotional_stability
            },
            "emotion_transitions": emotion_transitions,
            "emotion_cooccurrence": emotion_cooccurrence,
            "emotional_stability": emotional_stability,
            **text_features,  # Includes topic_analysis, linguistic_features, etc.
            "tags": text_features["tags"],
            "last_sentiment_updated_at": datetime.utcnow(),
            "bart_analysis": analyze_with_bart(text) if text else None,
            "sentiment_timeline": calculate_sentiment_timeline(sentence_analysis),
        }

        return analysis_result

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

def analyze_with_bart(text: str):
    try:
        inputs = bart_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bart_model(**inputs)
        probs = outputs.logits.softmax(dim=1)
        return {
            "entailment": float(probs[0][0]),
            "contradiction": float(probs[0][1]),
            "neutral": float(probs[0][2])
        }
    except Exception as e:
        print(f"Error in BART analysis: {str(e)}")
        return None

def calculate_sentiment_timeline(sentence_analysis):
    try:
        return [{
            "text": sent["text"],
            "sentiment": sent["vader_sentiment"]["compound"],
            "dominant_emotion": max(sent["emotion_scores"].items(), key=lambda x: x[1])[0]
        } for sent in sentence_analysis]
    except Exception as e:
        print(f"Error in sentiment timeline calculation: {str(e)}")
        return None

# Replace with this - only creates tables if they don't exist
def init_db():
    try:
        # Create tables only if they don't exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print("Database tables checked/created successfully")
    except Exception as e:
        print(f"Error checking database tables: {str(e)}")
        raise

# Call init_db() after all models are defined
init_db()

# Update your get_user_id endpoint
@app.post("/api/users/get-userid", response_model=Dict[str, str])
async def get_user_id(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        # Check if user exists
        user = db.query(User).filter(User.email == user_data.email).first()
        
        if user:
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            return {"user_id": str(user.user_id)}
        
        # If user doesn't exist, create a new one
        new_user = User(
            email=user_data.email,
            last_login=datetime.utcnow()
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"user_id": str(new_user.user_id)}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Update your analyze_text endpoint
@app.post("/api/analysis")
async def analyze_mood(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        user_email = data.get('email')
        content = data.get('content')
        day_rating = data.get('day_rating')
        selected_moods = data.get('selected_moods', [])
        
        # Get user
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            return JSONResponse(
                status_code=404,
                content={"error": "User not found. Please login first."}
            )

        # Analyze sentiment and emotions
        analysis_result = await analyze_sentiment(content, day_rating, selected_moods)
        
        # Get Gemini analysis
        gemini_analysis = get_gemini_analysis(content, analysis_result)  # Remove await
        
        # Create journal entry with default values and safe access
        journal_entry = MoodTracker(
            user_id=user.user_id,
            email=user_email,
            username=data.get('username', ''),
            user_journal=content,
            day_rating=day_rating,
            feelings=gemini_analysis,
            selected_moods=selected_moods,
            created_at=datetime.utcnow(),
            
            # Safe sentiment analysis with defaults
            sentiment={
                "scores": {
                    "neutral": 0.0,
                    "negative": 0.0,
                    "positive": 0.0
                },
                "analysis": "",
                "distribution": {
                    "labels": ["Positive", "Neutral", "Negative"],
                    "values": [0.0, 0.0, 0.0]
                }
            },
            
            # Safe emotion analysis with defaults
            roberta_emotions={
                "top_5": {},
                "all_emotions": {}
            },
            
            # Safe selected moods analysis
            selected_moods_analysis={
                "analysis": "",
                "selected_moods": selected_moods,
                "detected_emotions": {}
            },
            
            # Safe recommendations
            recommendations={
                "suggestions": [],
                "emotional_insight": ""
            },
            
            # Safe graph data
            graph_data={
                "sentiment_distribution": {
                    "labels": ["Positive", "Neutral", "Negative"],
                    "values": [0.0, 0.0, 0.0]
                },
                "emotion_distribution": {
                    "labels": [],
                    "values": []
                }
            },
            
            # Safe detailed analysis
            detailed_analysis={
                "sentence_level": [],
                "topic_analysis": {},
                "emotion_transitions": [],
                "emotion_cooccurrence": {},
                "emotional_stability": {
                    "variance": 0.0,
                    "stability_score": 0.0,
                    "emotional_shifts": 0
                }
            },
            
            # Additional safe fields
            emotion_transitions=[],
            emotional_stability={
                "variance": 0.0,
                "stability_score": 0.0,
                "emotional_shifts": 0
            },
            topic_analysis={}
        )
        
        # Enhanced graph_data structure with safe access
        graph_data = {
            "sentiment_distribution": {
                "labels": ["Positive", "Neutral", "Negative"],
                "values": [
                    analysis_result.get("sentiment", {}).get("scores", {}).get("positive", 0),
                    analysis_result.get("sentiment", {}).get("scores", {}).get("neutral", 0),
                    analysis_result.get("sentiment", {}).get("scores", {}).get("negative", 0)
                ]
            },
            "emotion_distribution": {
                "labels": list(analysis_result.get("roberta_emotions", {}).get("top_5", {}).keys()),
                "values": list(analysis_result.get("roberta_emotions", {}).get("top_5", {}).values())
            },
            "emotional_stability_metrics": {
                "labels": ["Stability Score", "Variance", "Emotional Shifts"],
                "values": [
                    analysis_result.get("emotional_stability", {}).get("stability_score", 0),
                    analysis_result.get("emotional_stability", {}).get("variance", 0),
                    analysis_result.get("emotional_stability", {}).get("emotional_shifts", 0)
                ]
            },
            "emotion_transitions": {
                "from_emotions": [t.get("from_emotion", "") for t in analysis_result.get("emotion_transitions", [])],
                "to_emotions": [t.get("to_emotion", "") for t in analysis_result.get("emotion_transitions", [])],
                "transition_scores": [t.get("transition_score", 0) for t in analysis_result.get("emotion_transitions", [])]
            },
            "temporal_analysis": {
                "morning_sentiment": analysis_result.get("detailed_analysis", {}).get("sentence_level", [{}])[0].get("vader_sentiment", {}).get("compound", 0),
                "evening_sentiment": analysis_result.get("detailed_analysis", {}).get("sentence_level", [{}])[-1].get("vader_sentiment", {}).get("compound", 0) if analysis_result.get("detailed_analysis", {}).get("sentence_level") else 0,
                "sentiment_progression": [
                    s.get("vader_sentiment", {}).get("compound", 0) 
                    for s in analysis_result.get("detailed_analysis", {}).get("sentence_level", [])
                ]
            },
            "emotion_intensity_timeline": {
                "timestamps": [i for i in range(len(analysis_result.get("detailed_analysis", {}).get("sentence_level", [])))],
                "intensities": [
                    max(s.get("emotion_scores", {}).values()) if s.get("emotion_scores") else 0
                    for s in analysis_result.get("detailed_analysis", {}).get("sentence_level", [])
                ]
            },
            "mood_correlation": {
                "selected_moods": selected_moods,  # Use selected_moods from request data
                "detected_emotions": list(analysis_result.get("roberta_emotions", {}).get("top_5", {}).keys()),
                "correlation_scores": [
                    analysis_result.get("roberta_emotions", {}).get("top_5", {}).get(emotion, 0)
                    for emotion in analysis_result.get("roberta_emotions", {}).get("top_5", {})
                ]
            },
            "emotional_balance": {
                "labels": ["Positive Emotions", "Negative Emotions", "Neutral Emotions"],
                "values": [
                    sum(1 for s in analysis_result.get("detailed_analysis", {}).get("sentence_level", []) 
                        if s.get("vader_sentiment", {}).get("compound", 0) > 0.05),
                    sum(1 for s in analysis_result.get("detailed_analysis", {}).get("sentence_level", []) 
                        if s.get("vader_sentiment", {}).get("compound", 0) < -0.05),
                    sum(1 for s in analysis_result.get("detailed_analysis", {}).get("sentence_level", []) 
                        if abs(s.get("vader_sentiment", {}).get("compound", 0)) <= 0.05)
                ]
            }
        }

        # Update journal_entry with new graph_data
        journal_entry.graph_data = graph_data
        
        # Update with actual analysis results if available
        if analysis_result:
            for key, value in analysis_result.items():
                if hasattr(journal_entry, key):
                    setattr(journal_entry, key, value)
        
        db.add(journal_entry)
        db.commit()
        db.refresh(journal_entry)
        
        # Return formatted response
        return JSONResponse(
            status_code=200,
            content={
                "id": journal_entry.id,
                "user_id": str(journal_entry.user_id),
                "email": journal_entry.email,
                "username": journal_entry.username,
                "journal_id": str(journal_entry.journal_id),
                "user_journal": journal_entry.user_journal,
                "day_rating": journal_entry.day_rating,
                "feelings": journal_entry.feelings,
                "selected_moods": journal_entry.selected_moods,
                "created_at": journal_entry.created_at.isoformat(),
                "updated_at": journal_entry.updated_at.isoformat() if journal_entry.updated_at else None,
                "sentiment": journal_entry.sentiment,
                "recommendations": journal_entry.recommendations,
                "roberta_emotions": journal_entry.roberta_emotions,
                "selected_moods_analysis": journal_entry.selected_moods_analysis,
                "detailed_analysis": journal_entry.detailed_analysis,
                "emotion_transitions": journal_entry.emotion_transitions,
                "emotion_cooccurrence": journal_entry.emotion_cooccurrence,
                "emotional_stability": journal_entry.emotional_stability,
                "topic_analysis": journal_entry.topic_analysis,
                "tags": journal_entry.tags,
                "archived": journal_entry.archived,
                "last_sentiment_updated_at": journal_entry.last_sentiment_updated_at.isoformat() if journal_entry.last_sentiment_updated_at else None,
                "bart_analysis": journal_entry.bart_analysis,
                "sentiment_timeline": journal_entry.sentiment_timeline,
                "emotion_intensity_metrics": journal_entry.emotion_intensity_metrics,
                "linguistic_features": journal_entry.linguistic_features,
                "behavioral_patterns": journal_entry.behavioral_patterns,
                "contextual_triggers": journal_entry.contextual_triggers,
                "social_references": journal_entry.social_references,
                "custom_tags": journal_entry.custom_tags,
                "graph_data": journal_entry.graph_data
            }
        )

    except Exception as e:
        db.rollback()
        print(f"Error in analyze_mood: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/analysis/{user_id}", response_model=List[AnalysisResponse])
async def get_user_analysis(user_id: str, db: Session = Depends(get_db)):
    try:
        # Convert string user_id to UUID
        user_uuid = uuid.UUID(user_id)
        
        analyses = db.query(MoodTracker).filter(
            MoodTracker.user_id == user_uuid
        ).order_by(MoodTracker.created_at.desc()).all()
        
        if not analyses:
            return []

        return [
            AnalysisResponse(
                id=analysis.id,
                journal_id=analysis.journal_id,
                username=analysis.username,
                day_rating=analysis.day_rating,
                feelings=analysis.feelings,
                selected_moods=analysis.selected_moods,
                created_at=analysis.created_at,
                updated_at=analysis.updated_at,
                sentiment=analysis.sentiment,
                roberta_emotions=analysis.roberta_emotions,
                selected_moods_analysis=analysis.selected_moods_analysis,
                recommendations=analysis.recommendations,
                graph_data={
                    "sentiment_distribution": analysis.sentiment.get("graph_data", {}).get("sentiment_distribution", {}),
                    "emotion_distribution": analysis.sentiment.get("graph_data", {}).get("emotion_distribution", {})
                },
                detailed_analysis=analysis.detailed_analysis,
                emotion_transitions=analysis.emotion_transitions,
                emotional_stability=analysis.emotional_stability,
                topic_analysis=analysis.topic_analysis
            ) for analysis in analyses
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test-db")
async def test_db(db: Session = Depends(get_db)):
    try:
        # Try to create a test entry
        test_analysis = MoodTracker(
            user_id=uuid.uuid4(),
            sentiment={"test": "data"},
            roberta_emotions={"test": "emotions"}
        )
        db.add(test_analysis)
        db.commit()
        db.refresh(test_analysis)
        
        # Clean up test data
        db.delete(test_analysis)
        db.commit()
        
        return {"status": "success", "message": "Database connection and operations working"}
    except Exception as e:
        print(f"Database test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the create_user endpoint
@app.post("/api/users/create")
async def create_user(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
        user_data = body.get('userData', {})
        email = user_data.get('email')

        if not email:
            return JSONResponse(
                status_code=400,
                content={"error": "Email is required"}
            )

        # Check if user exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            return JSONResponse(
                status_code=200,
                content={
                    "user_id": str(existing_user.user_id),
                    "message": "User already exists"
                }
            )

        # Generate a new UUID for the user
        new_user_id = uuid.uuid4()
        username = email.split('@')[0]  # Default username from email

        # Create new user with explicit UUID
        new_user = User(
            user_id=new_user_id,  # Explicitly set the UUID
            email=email,
            username=username,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        
        try:
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            print(f"Created new user: {email} with ID: {new_user_id}")  # Debug log
            
            return JSONResponse(
                status_code=201,
                content={
                    "user_id": str(new_user.user_id),
                    "email": new_user.email,
                    "username": new_user.username,
                    "message": "User created successfully"
                }
            )
        except Exception as db_error:
            db.rollback()
            print(f"Database error creating user: {str(db_error)}")  # Debug log
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(db_error)}"
            )
    
    except Exception as e:
        print(f"Error in create_user: {str(e)}")  # Debug log
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to create user: {str(e)}"}
        )

@app.get("/api/journals/{email}", response_model=List[AnalysisResponse])
async def get_user_journals(email: str, db: Session = Depends(get_db)):
    try:
        # Get user's entries
        entries = db.query(MoodTracker).filter(
            MoodTracker.email == email
        ).order_by(MoodTracker.created_at.desc()).all()
        
        if not entries:
            return []

        return [
            AnalysisResponse(
                id=entry.id,
                journal_id=entry.journal_id,
                username=entry.username,
                day_rating=entry.day_rating,
                feelings=entry.feelings,
                selected_moods=entry.selected_moods,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                sentiment=entry.sentiment,
                roberta_emotions=entry.roberta_emotions,
                selected_moods_analysis=entry.selected_moods_analysis,
                recommendations=entry.recommendations,
                graph_data={
                    "sentiment_distribution": entry.sentiment.get("graph_data", {}).get("sentiment_distribution", {}),
                    "emotion_distribution": entry.sentiment.get("graph_data", {}).get("emotion_distribution", {})
                },
                detailed_analysis=entry.detailed_analysis,
                emotion_transitions=entry.emotion_transitions,
                emotional_stability=entry.emotional_stability,
                topic_analysis=entry.topic_analysis
            ) for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Update get_current_user function
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token, 
            os.getenv("NEXTAUTH_SECRET"), 
            algorithms=["HS256"]
        )
        email: str = payload.get("sub")  # NextAuth uses 'sub' for email
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/api/journals/user/{user_id}")
async def get_user_journals(
    user_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100)  # Pagination
):
    try:
        user_uuid = uuid.UUID(user_id)
        
        # Optimized query with pagination
        journals = db.query(MoodTracker).filter(
            MoodTracker.user_id == user_uuid,
            MoodTracker.archived == False
        ).order_by(
            MoodTracker.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": journal.id,
                "journal_id": str(journal.journal_id),
                "user_id": str(journal.user_id),
                "email": journal.email,
                "username": journal.username,
                "user_journal": journal.user_journal,
                "day_rating": journal.day_rating,
                "feelings": journal.feelings,
                "selected_moods": journal.selected_moods or [],
                "created_at": journal.created_at.isoformat(),
                "updated_at": journal.updated_at.isoformat() if journal.updated_at else None,
                "sentiment": journal.sentiment or {},
                "recommendations": journal.recommendations or {},
                "roberta_emotions": journal.roberta_emotions or {},
                "selected_moods_analysis": journal.selected_moods_analysis or {},
                "detailed_analysis": journal.detailed_analysis or {
                    "emotional_stability": {
                        "emotional_shifts": 0
                    }
                },
                "emotion_transitions": journal.emotion_transitions or [],
                "emotion_cooccurrence": journal.emotion_cooccurrence or {},
                "emotional_stability": journal.emotional_stability or {},
                "topic_analysis": journal.topic_analysis or {},
                "tags": journal.tags or [],
                "archived": journal.archived,
                "last_sentiment_updated_at": journal.last_sentiment_updated_at.isoformat() if journal.last_sentiment_updated_at else None,
                "bart_analysis": journal.bart_analysis or {},
                "sentiment_timeline": journal.sentiment_timeline or [],
                "emotion_intensity_metrics": journal.emotion_intensity_metrics or {},
                "linguistic_features": journal.linguistic_features or {},
                "behavioral_patterns": journal.behavioral_patterns or {},
                "contextual_triggers": journal.contextual_triggers or {},
                "social_references": journal.social_references or {},
                "custom_tags": journal.custom_tags,
                "graph_data": journal.graph_data or {}
            }
            for journal in journals
        ]
    except Exception as e:
        print(f"Error fetching journals: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch journals: {str(e)}"
        )

def get_gemini_analysis(content: str, sentiment_data: dict = None) -> str:
    try:
        # Extract sentiment scores from the API data
        sentiment_scores = sentiment_data.get("sentiment", {}).get("scores", {})
        roberta_emotions = sentiment_data.get("roberta_emotions", {}).get("top_5", {})
        stability = sentiment_data.get("emotional_stability", {})
        
        # Format the analysis using actual data
        analysis = f"""The overall sentiment analysis reveals a complex emotional landscape. The emotional metrics show a {max(roberta_emotions.items(), key=lambda x: x[1])[0]} tone, with an interplay between positive ({sentiment_scores.get('positive', 0):.2%}) and negative ({sentiment_scores.get('negative', 0):.2%}) sentiments.
The dominant emotions detected are {', '.join(list(roberta_emotions.keys())[:3])}, with emotional stability score of {stability.get('stability_score', 0):.2f}. This pattern suggests {
'balanced emotions' if stability.get('stability_score', 0) > 0.7 
else 'some emotional fluctuation' if stability.get('stability_score', 0) > 0.4 
else 'significant emotional variation'}.

The emotional variance of {stability.get('variance', 0):.2f} and {stability.get('emotional_shifts', 0)} emotional shifts indicate {
'a very stable emotional state' if stability.get('emotional_shifts', 0) == 0
else 'some emotional transitions' if stability.get('emotional_shifts', 0) < 3
else 'frequent emotional changes'}."""

        return analysis
    except Exception as e:
        logging.error(f"Gemini analysis error: {str(e)}")
        return "Unable to generate emotional analysis."

# Add handler for serverless
handler = Mangum(app)

# Update the main block for local development only
if __name__ == "__main__":
    import uvicorn
    # Only run uvicorn in local development
    if not os.environ.get("VERCEL_ENV"):
        uvicorn.run(app, host="0.0.0.0", port=8000)
