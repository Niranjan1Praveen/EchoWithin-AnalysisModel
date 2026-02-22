# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import Counter
import numpy as np
import os
import gc
from supabase import create_client, Client
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Supabase configuration - ADD THESE TO ENVIRONMENT VARIABLES
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://hpumegppcvjhxgkavawh.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwdW1lZ3BwY3ZqaHhna2F2YXdoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk0MTE2MTcsImV4cCI6MjA4NDk4NzYxN30.JF_InceK-JR4LjVqebmNftlZGoXRM7Px-fQy_8mTu-c")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============ MEMORY OPTIMIZATIONS ============
# Set environment variables for memory optimization
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Force CPU and optimize memory
device = -1  # Force CPU
torch.set_num_threads(1)  # Limit CPU threads
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.5)  # Limit GPU memory if available

# Load ML models with memory optimizations
print("Loading emotion classifier model...")
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device,
    model_kwargs={
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32  # Use float32 for CPU
    },
    framework="pt"
)

print("Loading sentiment analysis model...")
sentiment_classifier = pipeline(
    "sentiment-analysis",
    device=device,
    model_kwargs={
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32
    },
    framework="pt"
)

# Enable garbage collection after model loading
gc.collect()
print("Models loaded successfully!")

# ============ HELPER FUNCTIONS ============
def clear_memory():
    """Helper function to clear memory after heavy operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def layer1_audio_processing(text):
    """Simulate audio analysis based on text patterns"""
    word_count = len(text.split())
    has_exclamation = '!' in text
    has_hesitation = any(word in text.lower() for word in ['uh', 'um', 'like', 'ah', 'erm'])
    has_pauses = '...' in text or '..' in text
    
    # Calculate speaking rate (simulated)
    words_per_minute = word_count * 60 / 5  # Assume 5 seconds to speak
    
    return {
        "energy": "High" if has_exclamation else "Medium" if word_count > 10 else "Low",
        "speech_pattern": "Hesitant" if has_hesitation else "Fluent",
        "speaking_rate": round(words_per_minute, 1),
        "pauses_detected": text.count('...') + text.count('..'),
        "voice_quality": "Emphatic" if has_exclamation else "Neutral"
    }

def layer2_semantic_analysis(text):
    """Analyze emotional content in text"""
    # Process text in smaller chunks if it's very long
    if len(text) > 512:  # Truncate long texts to save memory
        text = text[:512]
    
    emotions = emotion_classifier(text)[0]
    sentiment = sentiment_classifier(text)[0]
    
    # Extract key phrases (simplified)
    words = re.findall(r'\w+', text.lower())
    filtered_words = [w for w in words if len(w) > 3 and w not in ['this', 'that', 'with', 'from', 'have', 'what', 'about']]
    key_phrases = Counter(filtered_words).most_common(3)
    
    # Sentence structure analysis
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "primary_emotion": emotions['label'],
        "emotion_confidence": round(emotions['score'], 3),
        "sentiment": sentiment['label'],
        "sentiment_score": round(sentiment['score'], 3),
        "key_phrases": [phrase for phrase, _ in key_phrases],
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(np.mean([len(s.split()) for s in sentences]), 1) if sentences else 0
    }

def layer3_conversation_patterns(conversation):
    """Analyze patterns in the conversation"""
    # Extract only user messages
    user_messages = [msg for msg in conversation if msg['role'] == 'user']
    assistant_messages = [msg for msg in conversation if msg['role'] == 'assistant']
    
    if not user_messages:
        return {"error": "No user messages found"}
    
    # Calculate response times (simulated from text lengths)
    user_lengths = [len(msg['text'].split()) for msg in user_messages]
    assistant_lengths = [len(msg['text'].split()) for msg in assistant_messages]
    
    # Detect topic consistency (simple approach - word overlap)
    all_user_text = ' '.join([msg['text'] for msg in user_messages]).lower()
    user_words = set(re.findall(r'\w+', all_user_text))
    
    # Calculate engagement (based on message length progression)
    engagement_trend = "Increasing" if len(user_lengths) > 1 and user_lengths[-1] > user_lengths[0] else "Stable"
    
    return {
        "total_user_messages": len(user_messages),
        "avg_user_message_length": round(np.mean(user_lengths), 1) if user_lengths else 0,
        "avg_assistant_message_length": round(np.mean(assistant_lengths), 1) if assistant_lengths else 0,
        "message_length_trend": engagement_trend,
        "vocabulary_size": len(user_words),
        "unique_words_used": list(user_words)[:10]  # Sample of unique words
    }

def layer4_emotional_journey(conversation):
    """Track emotional changes throughout the conversation"""
    user_messages = [msg for msg in conversation if msg['role'] == 'user']
    
    if not user_messages:
        return {"error": "No user messages found"}
    
    emotional_journey = []
    emotions_over_time = []
    
    # Process messages one by one to avoid memory buildup
    for i, msg in enumerate(user_messages):
        text = msg['text']
        if len(text) > 512:  # Truncate long texts
            text = text[:512]
            
        emotion_result = emotion_classifier(text)[0]
        sentiment_result = sentiment_classifier(text)[0]
        
        emotional_journey.append({
            "message_index": i,
            "timestamp": msg.get('timestamp', i * 30000),  # Approx 30 seconds per message
            "emotion": emotion_result['label'],
            "emotion_score": round(emotion_result['score'], 3),
            "sentiment": sentiment_result['label'],
            "sentiment_score": round(sentiment_result['score'], 3),
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        })
        
        emotions_over_time.append(emotion_result['label'])
        
        # Clear memory periodically
        if i % 5 == 0:
            clear_memory()
    
    # Calculate overall metrics
    emotion_counts = Counter(emotions_over_time)
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Unknown"
    
    # Detect emotional shifts
    emotional_shifts = 0
    for i in range(1, len(emotions_over_time)):
        if emotions_over_time[i] != emotions_over_time[i-1]:
            emotional_shifts += 1
    
    return {
        "emotional_journey": emotional_journey,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": dict(emotion_counts),
        "emotional_shifts": emotional_shifts,
        "emotional_stability": "High" if emotional_shifts < len(user_messages)/3 else "Medium" if emotional_shifts < len(user_messages)/2 else "Low"
    }

# ============ ROUTES ============
@app.route('/api/analyze/<session_id>', methods=['GET'])
def analyze_conversation(session_id):
    """Fetch conversation from Supabase and analyze it"""
    try:
        # Clear memory before heavy operation
        clear_memory()
        
        # Fetch conversation from Supabase
        response = supabase.table('Conversation') \
            .select('*') \
            .eq('session_id', session_id) \
            .execute()
        
        if not response.data:
            return jsonify({"error": "Conversation not found"}), 404
        
        conversation_data = response.data[0]
        
        # Parse transcript JSON
        transcript = conversation_data.get('transcript_json', [])
        
        if not transcript:
            return jsonify({"error": "No transcript found"}), 400
        
        # Format conversation for analysis (ensure correct role format)
        formatted_conversation = []
        for turn in transcript:
            role = 'user' if turn.get('role') == 'user' else 'assistant'
            formatted_conversation.append({
                'role': role,
                'text': turn.get('text', ''),
                'timestamp': turn.get('timestamp', 0)
            })
        
        # Perform analysis (only on user messages)
        user_messages = [msg for msg in formatted_conversation if msg['role'] == 'user']
        
        # Process in batches to manage memory
        # Layer 1: Audio analysis for each user message (lightweight)
        layer1_results = []
        for msg in user_messages:
            layer1_results.append({
                "message_id": msg.get('timestamp', 0),
                "text": msg['text'][:100] + "..." if len(msg['text']) > 100 else msg['text'],
                "analysis": layer1_audio_processing(msg['text'])
            })
        
        # Layer 2: Semantic analysis for each user message (heavy)
        layer2_results = []
        for i, msg in enumerate(user_messages):
            layer2_results.append({
                "message_id": msg.get('timestamp', 0),
                "text": msg['text'][:100] + "..." if len(msg['text']) > 100 else msg['text'],
                "analysis": layer2_semantic_analysis(msg['text'])
            })
            # Clear memory every 5 messages
            if i % 5 == 0:
                clear_memory()
        
        # Layer 3: Conversation patterns (lightweight)
        layer3_results = layer3_conversation_patterns(formatted_conversation)
        
        # Layer 4: Emotional journey (heavy)
        layer4_results = layer4_emotional_journey(formatted_conversation)
        
        # Clear memory after heavy processing
        clear_memory()
        
        # Overall metrics
        overall_metrics = {
            "session_id": session_id,
            "total_user_messages": len(user_messages),
            "conversation_date": conversation_data.get('created_at'),
            "duration_seconds": conversation_data.get('total_duration', 0) // 1000,
            "dominant_emotion": layer4_results.get('dominant_emotion'),
            "emotional_shifts": layer4_results.get('emotional_shifts', 0),
            "avg_message_length": layer3_results.get('avg_user_message_length', 0),
            "vocabulary_richness": layer3_results.get('vocabulary_size', 0)
        }
        
        # Prepare final response
        analysis_result = {
            "success": True,
            "metadata": {
                "session_id": session_id,
                "user_id": conversation_data.get('user_id'),
                "user_name": conversation_data.get('user_name'),
                "created_at": conversation_data.get('created_at')
            },
            "overall_metrics": overall_metrics,
            "layers": {
                "layer1_audio": layer1_results,
                "layer2_semantic": layer2_results,
                "layer3_patterns": layer3_results,
                "layer4_journey": layer4_results
            },
            # Chart data for frontend
            "charts": {
                "emotion_timeline": [
                    {"index": i, "emotion": item['emotion'], "score": item['emotion_score']}
                    for i, item in enumerate(layer4_results.get('emotional_journey', []))
                ],
                "emotion_distribution": [
                    {"name": emotion, "value": count}
                    for emotion, count in layer4_results.get('emotion_distribution', {}).items()
                ],
                "message_lengths": [
                    {"message": i+1, "length": len(msg['text'].split())}
                    for i, msg in enumerate(user_messages)
                ]
            }
        }
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Clear memory on error
        clear_memory()
        return jsonify({"error": str(e)}), 500
    finally:
        # Always clear memory after request
        clear_memory()

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all available sessions for dropdown selection"""
    try:
        user_id = request.args.get('user_id')  # Optional filter
        
        query = supabase.table('Conversation') \
            .select('session_id, user_name, created_at, total_messages, total_duration') \
            .order('created_at', desc=True)
        
        if user_id:
            query = query.eq('user_id', user_id)
        
        response = query.execute()
        
        sessions = []
        for session in response.data:
            sessions.append({
                "session_id": session['session_id'],
                "user_name": session.get('user_name', 'Anonymous'),
                "date": session['created_at'],
                "message_count": session.get('total_messages', 0),
                "duration": session.get('total_duration', 0) // 1000  # Convert to seconds
            })
        
        return jsonify({"success": True, "sessions": sessions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Emotion Analysis API is running",
        "memory_optimized": True
    })

@app.before_request
def before_request():
    """Clear memory before each request"""
    clear_memory()

@app.after_request
def after_request(response):
    """Clear memory after each request"""
    clear_memory()
    return response

if __name__ == '__main__':
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    # Run with debug=False for production
    app.run(host='0.0.0.0', port=port, debug=False)