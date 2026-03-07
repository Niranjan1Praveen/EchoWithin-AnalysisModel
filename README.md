# EchoWithin-Analysis: Emotion Analysis Flask API

#### 🎓 Course – Happiness & Well-being (Academic Project)
#### 👥 Team Name – CODE4CHANGE
#### 💡 Project Theme – *Understanding Emotions Through Voice-Based AI*
#### 📩 Contact Email – niranjanpraveen@gmail.com

---

## 📋 Overview

**EchoWithin-Analysis** is a Python Flask microservice that powers the emotion interpretation engine for the EchoWithin platform. It provides a **4-layer emotional analysis** of conversations by processing both the content and patterns of user messages, returning rich insights for the frontend dashboard.

This service works in conjunction with the main Next.js application, fetching conversation data from Supabase and performing deep emotional analysis using transformer-based ML models.

---

## 🧠 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Next.js   │────▶│  Flask API   │────▶│  Supabase   │
│   Frontend  │◀────│  (Port 5000) │◀────│  Database   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  ML Models  │
                    │  (Hugging   │
                    │   Face)     │
                    └─────────────┘
```

---

## 🔬 Four-Layer Analysis Engine

### 🎵 Layer 1: Audio Processing (Simulated)
Analyzes vocal patterns from text as a proxy for audio features:
- **Energy Level**: High/Medium/Low based on text patterns
- **Speech Pattern**: Hesitant vs. Fluent (detects fillers like "um", "uh")
- **Speaking Rate**: Words per minute calculation
- **Pause Detection**: Counts ellipses and hesitations

### 📝 Layer 2: Semantic Analysis
Extracts emotional content from text using transformer models:
- **Primary Emotion**: joy, sadness, anger, fear, surprise, disgust
- **Emotion Confidence Score**: 0-1 scale
- **Sentiment**: Positive/Negative/Neutral
- **Key Phrases**: Most significant emotional words
- **Sentence Structure**: Length and complexity metrics

### 🔄 Layer 3: Conversational Patterns
Analyzes the dynamics of the conversation:
- **Message Count**: Total user messages
- **Average Length**: Words per message
- **Vocabulary Size**: Unique words used
- **Engagement Trend**: Increasing/Stable/Decreasing
- **Response Patterns**: User vs. Echo message lengths

### 📊 Layer 4: Emotional Journey
Tracks emotional changes throughout the conversation:
- **Emotional Timeline**: Per-message emotion tracking
- **Dominant Emotion**: Most frequent emotion
- **Emotion Distribution**: Percentage breakdown
- **Emotional Shifts**: Number of emotion changes
- **Emotional Stability**: High/Medium/Low rating

---

## 🚀 API Endpoints

### 1. List Available Sessions
```
GET /api/sessions
```
Returns all conversation sessions available for analysis.

**Response:**
```json
{
  "success": true,
  "sessions": [
    {
      "session_id": "session_001",
      "user_name": "Alex",
      "date": "2024-01-15T10:30:00Z",
      "message_count": 24,
      "duration": 930
    }
  ]
}
```

### 2. Analyze Specific Conversation
```
GET /api/analyze/<session_id>
```
Performs full 4-layer analysis on a specific conversation.

**Response:**
```json
{
  "success": true,
  "metadata": {
    "session_id": "session_001",
    "user_id": "user_123",
    "user_name": "Alex",
    "created_at": "2024-01-15T10:45:30Z"
  },
  "overall_metrics": {
    "total_user_messages": 12,
    "duration_seconds": 930,
    "dominant_emotion": "anxiety",
    "emotional_shifts": 3,
    "avg_message_length": 15.2,
    "vocabulary_richness": 87
  },
  "layers": {
    "layer1_audio": [...],
    "layer2_semantic": [...],
    "layer3_patterns": {...},
    "layer4_journey": {...}
  },
  "charts": {
    "emotion_timeline": [...],
    "emotion_distribution": [...],
    "message_lengths": [...]
  }
}
```

### 3. Health Check
```
GET /api/health
```
Verifies the API is running and can connect to Supabase.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Framework** | Flask 2.3.3 |
| **Database** | Supabase (PostgreSQL) |
| **ML Models** | Hugging Face Transformers |
| **Emotion Model** | `j-hartmann/emotion-english-distilroberta-base` |
| **Sentiment Model** | Hugging Face pipeline |
| **CORS** | Flask-CORS |
| **Environment** | python-dotenv |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- Supabase account and database
- Hugging Face account (optional, for custom models)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/echowithin-analysis.git
cd echowithin-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file:
```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=your_supabase_anon_key
FLASK_ENV=development
FLASK_APP=app.py
```

5. **Run the application**
```bash
python app.py
```
The server will start at `http://localhost:5000`

---

## 🔌 Integration with Next.js Frontend

In your Next.js application, fetch analysis data:

```typescript
// Example: Fetch analysis for a session
const fetchAnalysis = async (sessionId: string) => {
  const response = await fetch(`http://localhost:5000/api/analyze/${sessionId}`);
  const data = await response.json();
  
  if (data.success) {
    // Update your dashboard with analysis data
    setAnalysisData(data);
  }
};

// Example: Get available sessions
const fetchSessions = async () => {
  const response = await fetch('http://localhost:5000/api/sessions');
  const data = await response.json();
  
  if (data.success) {
    setSessions(data.sessions);
  }
};
```

---

## 📊 Sample Analysis Output

### Emotional Timeline Visualization
```javascript
{
  "emotion_timeline": [
    { "index": 0, "emotion": "anxiety", "score": 0.89 },
    { "index": 1, "emotion": "sadness", "score": 0.76 },
    { "index": 2, "emotion": "hope", "score": 0.82 }
  ]
}
```

### Emotion Distribution
```javascript
{
  "emotion_distribution": [
    { "name": "anxiety", "value": 5 },
    { "name": "sadness", "value": 3 },
    { "name": "hope", "value": 2 },
    { "name": "neutral", "value": 2 }
  ]
}
```

### Message Length Progression
```javascript
{
  "message_lengths": [
    { "message": 1, "length": 12 },
    { "message": 2, "length": 18 },
    { "message": 3, "length": 8 }
  ]
}
```

---

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Test individual endpoints:
```bash
# Health check
curl http://localhost:5000/api/health

# Get sessions
curl http://localhost:5000/api/sessions

# Analyze specific session
curl http://localhost:5000/api/analyze/session_001
```

---

## 🚢 Deployment

### Deploy to Railway (Recommended)

1. Create a `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  }
}
```

2. Set environment variables in Railway dashboard
3. Connect your GitHub repository
4. Deploy

### Deploy to Heroku

1. Create a `Procfile`:
```
web: gunicorn app:app
```

2. Set environment variables
3. Deploy using Heroku CLI or GitHub integration

---

## 🔧 Configuration

### CORS Settings
The API is configured to accept requests from your Next.js frontend. Update the `CORS` configuration in `app.py` if needed:

```python
CORS(app, origins=["http://localhost:3000", "https://yourdomain.com"])
```

### Model Selection
You can change the emotion model by updating:
```python
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base"
)
```

---

## 📈 Performance Considerations

- **Model Loading**: Models load once at startup and remain in memory
- **Caching**: Consider implementing Redis for frequently accessed analyses
- **Batch Processing**: For bulk analysis, implement async batch jobs
- **Rate Limiting**: Add rate limiting for production deployments

---

## 🔐 Security

- **API Keys**: Never commit `.env` files
- **CORS**: Restrict origins in production
- **Input Validation**: All inputs are validated before processing
- **Error Handling**: Graceful error messages without exposing internals

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is created for academic purposes as part of the Happiness & Well-being course.

---

## 👥 Team CODE4CHANGE

- **Niranjan Praveen** - Lead Developer
- Academic Mentor - [Name]
- Course: Happiness & Well-being

---

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Supabase for database infrastructure
- Vapi for voice AI technology
- ShadCN UI for component inspiration

---

## 📚 References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Supabase API](https://supabase.com/docs)
- [Emotion Models](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

---

> *"Understanding emotions through the power of voice and AI"*
