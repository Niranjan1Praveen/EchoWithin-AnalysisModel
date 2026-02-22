# ================================
# Production Memory-Safe app.py
# Same functionality as original
# ================================

import os
import gc
import re
import numpy as np
import warnings
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ================================
# MEMORY OPTIMIZATION SETTINGS
# ================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

app = Flask(__name__)
CORS(app)

# ================================
# SUPABASE CONFIG
# ================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# LAZY LOADED MODELS
# ================================

emotion_classifier = None
sentiment_classifier = None


def load_models():
    """
    Lazy load both models only once.
    Prevents OOM before port binding.
    """
    global emotion_classifier, sentiment_classifier

    if emotion_classifier is None:
        print("Loading emotion model...")
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1
        )
        gc.collect()

    if sentiment_classifier is None:
        print("Loading sentiment model...")
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            device=-1
        )
        gc.collect()


def clear_memory():
    gc.collect()


# ================================
# ANALYSIS LAYERS
# ================================

def layer1_audio_processing(text):
    word_count = len(text.split())
    has_exclamation = "!" in text
    has_hesitation = any(w in text.lower() for w in ["uh", "um", "ah", "erm", "like"])
    has_pauses = "..." in text or ".." in text

    words_per_minute = word_count * 12

    return {
        "energy": "High" if has_exclamation else "Medium" if word_count > 10 else "Low",
        "speech_pattern": "Hesitant" if has_hesitation else "Fluent",
        "speaking_rate": round(words_per_minute, 1),
        "pauses_detected": text.count("...") + text.count(".."),
        "voice_quality": "Emphatic" if has_exclamation else "Neutral"
    }


def layer2_semantic_analysis(text):
    load_models()

    text = text[:512]

    emotion = emotion_classifier(text)[0]
    sentiment = sentiment_classifier(text)[0]

    words = re.findall(r"\w+", text.lower())
    filtered = [w for w in words if len(w) > 3]
    key_phrases = Counter(filtered).most_common(3)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "primary_emotion": emotion["label"],
        "emotion_confidence": round(emotion["score"], 3),
        "sentiment": sentiment["label"],
        "sentiment_score": round(sentiment["score"], 3),
        "key_phrases": [k for k, _ in key_phrases],
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(
            np.mean([len(s.split()) for s in sentences]), 1
        ) if sentences else 0
    }


def layer3_conversation_patterns(conversation):
    user_msgs = [m for m in conversation if m["role"] == "user"]
    assistant_msgs = [m for m in conversation if m["role"] == "assistant"]

    if not user_msgs:
        return {"error": "No user messages found"}

    user_lengths = [len(m["text"].split()) for m in user_msgs]
    assistant_lengths = [len(m["text"].split()) for m in assistant_msgs]

    all_text = " ".join([m["text"] for m in user_msgs]).lower()
    vocab = set(re.findall(r"\w+", all_text))

    trend = "Increasing" if len(user_lengths) > 1 and user_lengths[-1] > user_lengths[0] else "Stable"

    return {
        "total_user_messages": len(user_msgs),
        "avg_user_message_length": round(np.mean(user_lengths), 1),
        "avg_assistant_message_length": round(np.mean(assistant_lengths), 1) if assistant_lengths else 0,
        "message_length_trend": trend,
        "vocabulary_size": len(vocab),
        "unique_words_used": list(vocab)[:10]
    }


def layer4_emotional_journey(conversation):
    load_models()

    user_msgs = [m for m in conversation if m["role"] == "user"]

    if not user_msgs:
        return {"error": "No user messages found"}

    emotions_over_time = []
    journey = []

    for i, msg in enumerate(user_msgs):
        text = msg["text"][:512]

        emotion = emotion_classifier(text)[0]
        sentiment = sentiment_classifier(text)[0]

        emotions_over_time.append(emotion["label"])

        journey.append({
            "message_index": i,
            "timestamp": msg.get("timestamp", i * 30000),
            "emotion": emotion["label"],
            "emotion_score": round(emotion["score"], 3),
            "sentiment": sentiment["label"],
            "sentiment_score": round(sentiment["score"], 3),
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        })

        if i % 3 == 0:
            clear_memory()

    counts = Counter(emotions_over_time)
    dominant = counts.most_common(1)[0][0] if counts else "Unknown"

    shifts = sum(
        1 for i in range(1, len(emotions_over_time))
        if emotions_over_time[i] != emotions_over_time[i - 1]
    )

    return {
        "emotional_journey": journey,
        "dominant_emotion": dominant,
        "emotion_distribution": dict(counts),
        "emotional_shifts": shifts,
        "emotional_stability": (
            "High" if shifts < len(user_msgs)/3 else
            "Medium" if shifts < len(user_msgs)/2 else
            "Low"
        )
    }


# ================================
# ROUTES
# ================================

@app.route("/api/analyze/<session_id>", methods=["GET"])
def analyze_conversation(session_id):
    try:
        clear_memory()

        response = supabase.table("Conversation") \
            .select("*") \
            .eq("session_id", session_id) \
            .execute()

        if not response.data:
            return jsonify({"error": "Conversation not found"}), 404

        conversation_data = response.data[0]
        transcript = conversation_data.get("transcript_json", [])

        if not transcript:
            return jsonify({"error": "No transcript found"}), 400

        formatted = [
            {
                "role": "user" if t.get("role") == "user" else "assistant",
                "text": t.get("text", ""),
                "timestamp": t.get("timestamp", 0)
            }
            for t in transcript
        ]

        user_msgs = [m for m in formatted if m["role"] == "user"]

        layer1 = [
            {
                "message_id": m["timestamp"],
                "text": m["text"][:100],
                "analysis": layer1_audio_processing(m["text"])
            }
            for m in user_msgs
        ]

        layer2 = [
            {
                "message_id": m["timestamp"],
                "text": m["text"][:100],
                "analysis": layer2_semantic_analysis(m["text"])
            }
            for m in user_msgs
        ]

        layer3 = layer3_conversation_patterns(formatted)
        layer4 = layer4_emotional_journey(formatted)

        overall_metrics = {
            "session_id": session_id,
            "total_user_messages": len(user_msgs),
            "conversation_date": conversation_data.get("created_at"),
            "duration_seconds": conversation_data.get("total_duration", 0) // 1000,
            "dominant_emotion": layer4.get("dominant_emotion"),
            "emotional_shifts": layer4.get("emotional_shifts", 0),
            "avg_message_length": layer3.get("avg_user_message_length", 0),
            "vocabulary_richness": layer3.get("vocabulary_size", 0)
        }

        return jsonify({
            "success": True,
            "metadata": {
                "session_id": session_id,
                "user_id": conversation_data.get("user_id"),
                "user_name": conversation_data.get("user_name"),
                "created_at": conversation_data.get("created_at")
            },
            "overall_metrics": overall_metrics,
            "layers": {
                "layer1_audio": layer1,
                "layer2_semantic": layer2,
                "layer3_patterns": layer3,
                "layer4_journey": layer4
            },
            "charts": {
                "emotion_timeline": [
                    {"index": i, "emotion": j["emotion"], "score": j["emotion_score"]}
                    for i, j in enumerate(layer4.get("emotional_journey", []))
                ],
                "emotion_distribution": [
                    {"name": e, "value": c}
                    for e, c in layer4.get("emotion_distribution", {}).items()
                ],
                "message_lengths": [
                    {"message": i + 1, "length": len(m["text"].split())}
                    for i, m in enumerate(user_msgs)
                ]
            }
        })

    except Exception as e:
        clear_memory()
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    try:
        user_id = request.args.get("user_id")

        query = supabase.table("Conversation") \
            .select("session_id, user_name, created_at, total_messages, total_duration") \
            .order("created_at", desc=True)

        if user_id:
            query = query.eq("user_id", user_id)

        response = query.execute()

        sessions = [
            {
                "session_id": s["session_id"],
                "user_name": s.get("user_name", "Anonymous"),
                "date": s["created_at"],
                "message_count": s.get("total_messages", 0),
                "duration": s.get("total_duration", 0) // 1000
            }
            for s in response.data
        ]

        return jsonify({"success": True, "sessions": sessions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "full_functionality_enabled": True
    })


# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)