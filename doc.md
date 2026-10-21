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


# Train Whisper Medium for Marathi + Hindi on Google Colab T4

Yes. You can use Option B — Whisper Medium on your 15 GB T4 GPU to build a multilingual speech-to-text model specifically optimized for Marathi and Hindi.

For your Tetrax project, I recommend building a model that accepts Marathi or Hindi speech and returns text in the same language. You can later expose it as a paid API service.

One important change: Svarah should not be your primary training dataset. It contains English speech from Indian speakers, not Marathi- and Hindi-language transcripts. You need datasets containing actual Marathi and Hindi audio paired with transcripts.

## 1. What you can achieve

[Design of a Handwritten Devanagari Typeface – Kimya Gandhi. किमया गांधी](https://images.openai.com/static-rsc-4/UZngl2U1bXRYhiUhPAtwJWAsjOl1kgx-9qEsa2YD3OxsvbRgAZsVCxu_1jRYYF9hd182TW-kJ9elA9hTDhG9hyyv59hjCMp8QGISqkXstLYQquYLB1h4fCH8SilD6F-1hoCEY5Z8ylH4-lWCYd2qu3X7zO2UkQzsm3nhNtij8-g?purpose=inline)

Marathi Speech-to-Text

Convert spoken Marathi into Marathi Devanagari text, including everyday conversations and recordings represented in your training data.

[Hindi, Marathi and  Nepali Fonts: South Asian Language Resource Center](https://images.openai.com/static-rsc-4/VbJ5F2XaPUPW5DH-3mqcvq0JEue19nxAGrmhQnlL1bfR6g7NURdbmUbpQPSgLiEMFnfR1tYyHpvca4FjUTZ0DvUGfHWsiu7wXkfZTDkS--iK_xUD9xUuxcHXeHw-xEplnJtQiP5pTIM194jdy9FEEWZDfRKLxRdjhL_ZaP0jOnQ?purpose=inline)

Hindi Speech-to-Text

Transcribe Hindi speech into Hindi Devanagari text, including conversational audio represented in your training data.

[Real-Time Hindi Meeting Transcription with Tactiq](https://images.openai.com/static-rsc-4/fp7H8BEBbjTDq5xbAw6RhWQzd-61kVhq58NWd4KAHuFdVbhv5Fs_w8Hg5Q_dK_bvL3UFhRMqrE3cnX28ThCooxzQZXgaHqe3QqavUbexiOj8L9ubUb9cr_t94pHHP9ppvrTDJcD5UDvWTEJjFBSPPqKTELk9EvmVpdyeiqJrDPE?purpose=inline)

One bilingual model

A single fine-tuned Whisper Medium checkpoint can recognize both languages. You do not necessarily need two separate models.

You can aim for:

- File-based transcription through a REST API.
- Automatic language detection, or an explicit `language=mr` / `language=hi` parameter.
- Timestamped transcripts and subtitle generation.
- A later WebSocket service for live captions.

Fine-tuning improves performance only if your data, evaluation and training setup support it. It does not guarantee better accuracy than the original Whisper model.

## 2. Which datasets should you use?

I recommend combining a primary training dataset with an independent evaluation dataset.

[AI4Bharat](https://images.openai.com/static-rsc-4/sjfCt819_o2QTOOg4d1eXcuDe4K3T1--UMBr3AV6J6lcd3sVnSjFAAz9_wxRvAorKW63XxU4eol9htx4pmcGjfjNMyqzoVrqqS_LwxHZcmcnRXdDwd5L2uDdsCIFnOWAfaPnVoJpIgw1YXkXUZkj4Cas5TdLidkb-i5VpeyPFnc?purpose=inline)

### Dataset 1 — IndicVoices

Primary training dataset

Marathi + Hindi · Audio with transcripts

My first choice. It contains natural, spontaneous and conversational Indian speech. The published dataset includes separate Marathi and Hindi configurations and training/validation splits. Access requires accepting the dataset conditions and sharing contact information.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

+1



[Open IndicVoices on Hugging Face](https://huggingface.co/datasets/ai4bharat/IndicVoices?utm_source=chatgpt.com)

[Common Voice: Mozilla veröffentlicht den bisher größten öffentlichen transkribierten Sprachdatensatz – Mozilla Presse-Center](https://images.openai.com/static-rsc-4/_WUn9G39XeAxBms9PW53b6ruW45JPGXyiDu0qda7G8jQlEE98nqXySrgt1PTtu1Sq0wfmpYMJgBSKEpwiOPurBG7RdYdBMpzBdGFkYjuYIlKtj28MMT1_eTKgaazeHVQSjoLrXlqbvhbauC_UMe6tujr19hwP_IoRElOAtHAb-A?purpose=inline)

### Dataset 2 — Common Voice

Additional training data or evaluation

Use the Hindi and Marathi releases if their current availability, licensing, transcripts and audio quality suit your project. This can broaden speaker and recording diversity.

[Open Mozilla Common Voice](https://commonvoice.mozilla.org/?utm_source=chatgpt.com)

[google/fleurs · Datasets at Hugging Face](https://images.openai.com/static-rsc-4/wTi8V5N5gCayLZ9lnInTVuy2G-Grz-fabkUTeQYDxWv_ehH-Gut4fF0piOK87e13WxKEeKJN94g2B5Nn_Dqy6cKpYMyCC24QjtVY6sDRZN1BPILfzWZNXpypbHXeT8pooq3imHWtmqsiLsY0HkhsQ5IIbfzP-HUmd-S7nf4f1DQ?purpose=inline)

### Dataset 3 — FLEURS

Independent evaluation

Use the Hindi and Marathi subsets to measure recognition quality on a consistent multilingual benchmark. Keep this data separate from your training set.

[Open FLEURS on Hugging Face](https://huggingface.co/datasets/google/fleurs?utm_source=chatgpt.com)

### How much data do you need?

For a first experiment on your T4, use a manageable subset rather than downloading every recording.

| Experiment                 | Marathi                          | Hindi                            | Purpose                                    |
| -------------------------- | -------------------------------- | -------------------------------- | ------------------------------------------ |
| Pipeline smoke test        | 100–300 clips                    | 100–300 clips                    | Confirm preprocessing and training work    |
| Initial LoRA experiment    | 1–5 hours                        | 1–5 hours                        | Test whether fine-tuning helps             |
| More meaningful experiment | 10–50+ hours                     | 10–50+ hours                     | Improve coverage and assess generalization |
| Production-grade model     | As much quality data as feasible | As much quality data as feasible | Optimize against real customer audio       |

These are practical starting targets, not guaranteed thresholds for good accuracy. Dataset diversity, transcript quality, duration distribution and speaker coverage matter as much as raw hours.

## 3. How to load Marathi and Hindi data in Colab

The IndicVoices repository reports approximately 745 GB across its full release. You do not need all of that: load only Marathi and Hindi, and begin with a small experiment.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Datasets at Hugging Face

+1



First, accept the dataset conditions on Hugging Face and authenticate in Colab.

```
!pip install -U datasets huggingface_hub
```

```
from huggingface_hub import notebook_loginnotebook_login()
```

Then inspect the Marathi dataset:

```
from datasets import load_datasetmarathi = load_dataset(    "ai4bharat/IndicVoices",    "marathi",    split="train",    streaming=True)sample = next(iter(marathi))print("Columns:", sample.keys())for key, value in sample.items():    if key not in ("audio_filepath", "audio"):        print(key, str(value)[:200])
```

Inspect Hindi in the same way:

```
hindi = load_dataset(    "ai4bharat/IndicVoices",    "hindi",    split="train",    streaming=True)print(next(iter(hindi)).keys())
```

Streaming lets you inspect examples without downloading the entire split first. Before training, verify the audio and transcript fields, durations, language labels and available metadata against the current dataset schema. Streaming is useful for inspection; a shuffled, bounded, reproducible training subset or a properly prepared local dataset is generally easier to use for controlled experiments.

## 4. Which model should you train?

Use the multilingual checkpoint, not the English-only version.

Recommended base checkpoint

## openai/whisper-medium

769 million parameters · Multilingual ASR · Supports Hindi and Marathi

[Model card and documentation](https://huggingface.co/openai/whisper-medium?utm_source=chatgpt.com)

Initial training settings

- GPU: Colab NVIDIA T4, 15 GB VRAM
- Fine-tuning: LoRA / PEFT
- Precision: FP16
- Batch size: 1
- Gradient accumulation: 8
- Gradient checkpointing: enabled
- Initial training: a few hundred examples, then a small multi-hour subset
- Save checkpoints and logs to Google Drive

The exact settings may need adjustment to avoid CUDA out-of-memory errors. Validate the training loop before scaling up.

For one bilingual model, use the appropriate language prompt for each sample: `marathi` for Marathi and `hindi` for Hindi, with the transcription task set to `transcribe`. Preserve the language identity in your dataset manifest so you can evaluate both languages separately.

## 5. How to measure whether your model is improving

Track these metrics independently for Marathi and Hindi:

- WER (Word Error Rate): lower is better.
- CER (Character Error Rate): useful for diagnosing errors in Devanagari text.
- Inference speed: seconds required to transcribe a given duration of audio.
- Generalization: results on unseen speakers and recordings.

Compare three systems: original Whisper Medium, fine-tuned Whisper Medium, and—if practical—Whisper Small fine-tuned on the same data. This will tell you whether the larger model's accuracy justifies its higher serving cost.

## 6. My final recommendation for Tetrax

Start with one multilingual Whisper Medium model, fine-tuned on separate Marathi and Hindi training samples from IndicVoices. Use Common Voice as an additional data source if needed, and FLEURS or another genuinely held-out dataset for evaluation.

Once the model performs reliably, deploy it behind a FastAPI service with language selection, audio upload, transcription output, API keys and usage metering.

The key next step is to verify that IndicVoices loads successfully in your Colab and inspect its actual audio/transcript schema before writing the training notebook. That avoids wasting your limited GPU session on a dataset-preparation error.