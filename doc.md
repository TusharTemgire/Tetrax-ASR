# Build your own AI API service using AI4Bharat Svarah

The AI4Bharat Svarah dataset can be a starting point for building an Indian-language speech AI service under your Tetrax brand. The key is to choose a service that the dataset can genuinely support, fine-tune an appropriate pretrained model, and expose it through a production-ready API.

My recommendation is to start with a Speech-to-Text (STT) API focused on Indian English and Indian accents, then expand into multilingual transcription, subtitles, and real-time meeting captions if the dataset and evaluation results support those use cases.

## 1. What exactly is in Svarah?

# 9.6 hours

Transcribed English speech

# 117

Speakers

# 19 states

Geographic diversity

The dataset contains 6,656 audio examples with English transcripts, including read speech and spontaneous conversational speech. Speakers have diverse Indian accents and native-language backgrounds. The published Hugging Face dataset exposes a `test` split, not a ready-made training split.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

+2



Important distinction: Svarah contains English speech from speakers with diverse Indian-language backgrounds. It is not a dataset of transcribed Marathi, Hindi, Tamil and other Indian-language speech.

You can use it to improve and evaluate recognition of Indian-accented English, but it is too small by itself to train a competitive speech model from scratch.

## 2. Which API services could you build?

[Faster Whisper – Fast and Accurate AI Speech-to-Text Software](https://images.openai.com/static-rsc-4/mYruNMIonYC4apEGsyFIctQlEGL6IXGJlK96KrC4BeWZCAeQCJaw0rUzC1btd1ijUjE5n52SW2aAZRHPIGTl42c15xKWrKE-oUE20_faY8VM0VWBvuM3XiPzETWwDlzR9Ilm_1UBQFX9v0dmJvmdTD-AzBLuqm4j10ixbZJUeeY?purpose=inline)

### 1. Tetrax Speech-to-Text API

Best starting point

Convert uploaded audio into English text, optimized for Indian accents.

- REST API for WAV, MP3 and other supported audio formats
- Timestamped transcription and optional word-level timestamps
- Vocabulary and domain-specific transcription options
- Usage-based billing per audio minute

[Soundcribe - audio transcription platform by ckuxdesign on Dribbble](https://images.openai.com/static-rsc-4/o_XV6KecI6bQdgAHYV99zdZ7KOaHU6WkiplZZvWUEZ2HTLrgp8eoOrbJSvW6NziWcDwC9R5o-cYugONOludgyeVCKraXk3c6abqgb9-n9BP5sb4s9aLFPbCMnGYlGXeVZtv0VG5p-PSx76qD9Z_AOtlvOMMI5RQlwdVl3w880zI?purpose=inline)

### 2. Real-time transcription API

Stream microphone audio to your backend and return live captions over WebSocket.

Potential customers include accessibility applications, online meetings, education platforms and collaboration software.

Svarah can help with accent robustness, but you will also need streaming-specific engineering and additional conversational data.

[A Guide to Call Center Sentiment Analysis Best Practices](https://images.openai.com/static-rsc-4/THf9TVi_v4dUJCPYW1Mg_pqToFsl7K2EY3eqdypkVspcAXohZtqUoXUY_EKq_BasHCBau_XRFC89C8Y-Opz6vLgp2nPNZeMqibNMo27zr-xAodFFR8BMBEnqt6rSrq09tW9ER9m8keGY31vdRW6G_G9Sg1Shmg_A2jrNjkdQAGU?purpose=inline)

### 3. Indian-English call transcription API

Build a service for customer-support calls, sales conversations and business recordings.

Possible features include searchable transcripts, custom vocabulary, conversation summaries and analytics.

For telephone audio, gather additional 8 kHz telephony recordings and domain-specific transcripts.

[MAI-Transcribe-2-Streaming 對比 Gemini 3.5 Transcribe Live](https://images.openai.com/static-rsc-4/NUHVB6TVHot6AQAYKHSQexJMSWUXrTaxIEr5_JlTg_eCc3MS3VL4PN0Fc-9xsweyPqheO3vRSN1x18r-IRDMvQE-KHh9c2IiH9-7NClZXlytbTjBAXihQuRt7h3Tyu-o-Fc_tD1bj4v6GJJEDWLMAmjOtmtEj9cUuXQwUFVfXPk?purpose=inline)

### 4. Indian Accent ASR Benchmark API

Offer an API or dashboard to compare speech-recognition systems on Indian-accented English.

This is useful for research teams and companies choosing between open-source models and hosted speech APIs. Svarah is particularly well suited to evaluation.

My recommendation is to start with Service 1, then add Service 2 as a separate streaming interface over the same transcription engine.

## 3. Which model should you fine-tune?

These are the most practical candidates for your first version.

[OpenAI ChatGPT & Whisper API Now Live | Start Building with New Tools](https://images.openai.com/static-rsc-4/aNZNF2VogMJ_iWfROxsONyMXUI_KMmh9B5oOUMZD9VmjCoWU_ORk2C8YyyjL5WA2NKOyhKFvhUzEXKB_22H6yCVuIHsRoZ842muQWFzsqg1OY6hk83X3YkwfmRA1oHFPCK2Mce148WThQzKgrZzSM07cHxMRpdUi3DTqHW78x-U?purpose=inline)

### Option A — Whisper Small

Recommended first experiment

Approximately 244 million parameters

Fine-tune openai/whisper-small on suitable Indian-English training data.

Why: A practical balance between recognition quality, GPU requirements and deployment cost. Use Svarah for held-out evaluation rather than training and testing on the same examples.

[Whisper: Robust Multilingual Speech Recognition via Large-Scale Weak S](https://images.openai.com/static-rsc-4/d-EyhF9ppuNX8K945LeBF6o1QpOZTbxHEC54ylempICyoS2LFrlHIedl9vESjt6NPM0myxzz88-FagXA2dCsgT24MxrUg8WE3qn2cxe4nCCJy5cAf1PUBCUnmVMLV1edfA0Lrw3Xvj0Y-tTO2vmPJ3pr_-PCSq8Q8FACaeOJgzI?purpose=inline)

### Option B — Whisper Medium

Approximately 769 million parameters

Fine-tune openai/whisper-medium when you have sufficient GPU memory and want to compare a larger model.

Why: A useful accuracy benchmark, but it has higher inference and hosting costs than Small.

[음성인식 AI는 어떻게 말을 알아들을까?](https://images.openai.com/static-rsc-4/4fSRlJuvYNBIH2M_i7J90QJp40rI1PkViDyEplxZwkuEm_BkeUkN4o20bJTyBPqF03HW8NB8OZofDo0K_33Tx92LOdlspSo2usskAXuOk6gtpgHygKVdbVDA3EkPPejIhMK3rz0KiL1qVP5UEiV3M0RInFgyYF78LhW87xMbyoY?purpose=inline)

### Option C — Indic-focused ASR

Evaluate an AI4Bharat IndicWhisper or another suitable Indic ASR checkpoint if you plan to expand beyond English.

Why: Better aligned with a future multilingual product, but Svarah alone cannot teach the model to transcribe Indian-language speech. You will need appropriately transcribed multilingual data.

The original Svarah paper reported word error rates (WER) of 13.6% for Whisper Base, 8.3% for Whisper Medium and 7.2% for Whisper Large on its benchmark. Lower WER is better. These are published baseline results, not guaranteed results for a fine-tuned Tetrax model.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



Recommended model stack

- Training: Whisper Small with Hugging Face Transformers and parameter-efficient fine-tuning if appropriate.
- Inference: Transformers initially; benchmark `faster-whisper` or another optimized inference runtime before production.
- Backend: Python + FastAPI.
- Streaming: WebSocket, with a streaming-capable recognition pipeline.
- Frontend: Next.js + React.
- Infrastructure: GPU inference initially, with a CPU fallback for lightweight workloads.

## 4. The training strategy I recommend

The biggest issue is that Svarah is published as a benchmark dataset with a `test` split. Don't fine-tune on the entire dataset and then report performance on those same recordings as proof of accuracy.

Instead, follow this workflow:

Stage 1 — Establish a baseline

Run a pretrained Whisper Small model against Svarah and calculate WER.

Stage 2 — Build training data

Collect additional Indian-English speech with reliable transcripts and speaker-separated train/validation/test splits.

Stage 3 — Fine-tune Whisper Small

Train using the new training set, validate checkpoints and tune hyperparameters.

Stage 4 — Evaluate independently

Compare the baseline and fine-tuned model on untouched test data, including Svarah where appropriate.

Stage 5 — Deploy Tetrax API

Serve the selected checkpoint with authentication, usage tracking, rate limits and monitoring.

You can also use Svarah as a benchmark without fine-tuning it at all. This lets you compare models before investing in GPU training.

## 5. Design your own API service

Here is a practical initial API specification for Tetrax.

| Endpoint                        | Purpose                                     |
| ------------------------------- | ------------------------------------------- |
| `POST /v1/audio/transcriptions` | Transcribe an uploaded audio file           |
| `POST /v1/audio/jobs`           | Start a long-running transcription job      |
| `GET /v1/audio/jobs/{job_id}`   | Check job status and retrieve results       |
| `WS /v1/audio/stream`           | Send audio chunks and receive live captions |
| `GET /v1/models`                | List available recognition models           |
| `GET /v1/usage`                 | Retrieve usage and billing information      |

Example request:

```
curl -X POST \
  https://api.tetrax.in/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@meeting.wav" \
  -F "model=tetrax-stt-indian-en"
```

Example response:

```
{
  "id": "tr_123456",
  "model": "tetrax-stt-indian-en",
  "language": "en",
  "text": "Welcome to today's project meeting.",
  "duration": 4.2,
  "usage": {
    "audio_seconds": 4.2
  }
}
```

This is an example of the response format you could implement, not a live API.

### Suggested project structure

```
tetrax-stt/
├── training/
│   ├── prepare_dataset.py
│   ├── train_whisper.py
│   ├── evaluate.py
│   └── configs/
├── api/
│   ├── main.py
│   ├── routes/
│   │   ├── transcriptions.py
│   │   ├── jobs.py
│   │   └── models.py
│   ├── services/
│   │   ├── inference.py
│   │   └── audio_processing.py
│   └── schemas/
├── tests/
├── requirements.txt
└── README.md
```

For production, add a job queue, private object storage, API-key management, metering and model versioning. Keep GPU inference workers separate from the API process so that request handling doesn't block while audio is being transcribed.

## 6. How to make the service commercially useful

You will need to compete on a measurable advantage, not simply on having your own model.

Accuracy on Indian accents

Publish reproducible WER results and compare against a pretrained baseline and selected competing services.

Cost per audio minute

Measure GPU utilization, throughput, model startup time and cost per hour of audio processed before setting prices.

Real-time capability

Add partial transcripts, timestamps, reconnect handling and streaming latency measurements.

Production reliability

Add tenant isolation, request limits, secure audio retention policies, usage quotas and monitoring.

### What about Marathi and other Indian languages?

Svarah is a starting point for Indian-accented English. To build a genuinely multilingual Tetrax API, combine it with suitable multilingual speech datasets such as AI4Bharat IndicVoices and evaluate an appropriate multilingual ASR model.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://ai4bharat.iitm.ac.in\&sz=32)

AI4Bharat

+1



Also, the Svarah dataset is released under CC BY 4.0. Follow its attribution requirements and check the separate license terms of every pretrained model and additional dataset you use.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

+1



## 7. Your next step

I recommend building Tetrax STT v1: Indian-English Speech Recognition API, with Whisper Small as the first candidate and Whisper Medium as the comparison model.

To turn this into a concrete implementation, select your preferred starting approach:

1\. Training approach

Evaluate pretrained models first (recommended)

Fine-tune Whisper Small using additional training data

Build a complete ASR benchmark and comparison suite

2\. Training hardware

Cloud GPU (recommended)

Local Windows PC

3\. Initial product

REST transcription API only

Real-time streaming transcription API

Both REST and streaming APIs

Build my implementation plan