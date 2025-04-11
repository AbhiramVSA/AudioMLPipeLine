

from groq import Groq

import streamlit as st
import asyncio
import re
import base64
from io import BytesIO
import tempfile

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.gemini import GeminiModel

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from load_api import Settings

import nest_asyncio
import asyncio
nest_asyncio.apply()

async def transcribe_audio(audio_path: str):
    """Transcribes audio using Groq's Whisper model properly."""
    client = Groq(api_key=settings.GROQ_API_KEY)
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(  # ✅ Run synchronously
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    responses = sanitize_transcript(response.text)
    return responses


settings = Settings()

# ---------------- Whisper Local Model Transcriber ---------------- #
# def initWhisper(audio_path: str) -> str:
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     model_id = "openai/whisper-large-v3"

#     model = AutoModelForSpeechSeq2Seq.from_pretrained(
#         model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#     )
#     model.to(device)

#     processor = AutoProcessor.from_pretrained(model_id)

#     pipe = pipeline(
#         "automatic-speech-recognition",
#         model=model,
#         tokenizer=processor.tokenizer,
#         feature_extractor=processor.feature_extractor,
#         torch_dtype=torch_dtype,
#         device=device,
#     )

#     result = pipe(audio_path, return_timestamps=True)
#     return result["text"]

# ---------------- Sanitization ---------------- #
def sanitize_transcript(transcript: str) -> str:
    card_pattern = r'\b((?:\d[ -]?){6})(?:(?:[Xx\- ]{1,6}|\d[ -]?){2,9})([ -]?\d{4})\b'

    def mask_card(match):
        first_part = re.sub(r'[^0-9]', '', match.group(1))[:6]
        last_part = re.sub(r'[^0-9]', '', match.group(2))[-4:]
        return f"{first_part}{'X' * max(0, 16 - len(first_part) - len(last_part))}{last_part}"

    sanitized = re.sub(card_pattern, mask_card, transcript)
    ssn_pattern = r'\b(\d{3}|X{3})[- ]?(\d{2}|X{2})[- ]?(\d{4})\b'
    sanitized = re.sub(ssn_pattern, r'XXX-XX-\3', sanitized)
    return sanitized

# ---------------- Gemini Agent for Report ---------------- #
model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=settings.GROQ_API_KEY))
agent = Agent(
    model,
    system_prompt='''
---

You will be provided with a transcript of a customer support conversation.  
Your task is to extract relevant details and generate a **structured, professional Customer Support Report** in **Markdown format**, following the instructions below.

---

### **Extraction Rules:**

- **Strictly extract only what is explicitly stated** in the transcript.
- **Do not assume, infer, or fabricate** any missing details.
- If a field is missing or unclear, use **"Not specified"** or **"[Unclear from transcript]"**.
- Mask all sensitive information as described.
- Maintain a **clean and readable Markdown layout** using appropriate headers, bold text, and tables where needed.

---

### **Prohibited Actions:**

- ❌ Never invent:
  - Dates or times
  - Customer demographics (e.g., age, location)
  - Business names or merchant details
  - Resolution timelines not mentioned

- ❌ Avoid:
  - Hallucinating resolutions
  - Assuming transaction context
  - Adding extra analysis

---

### **Markdown Output Format:**

```markdown
# Customer Support Report

---

## 1. Customer Information
- **Full Name:** [First Last]
- **Age:** [XX] (Category: [Young/Adult/Elderly])
- **Locality:** [If mentioned]
- **Account Number:** [#######]

---

## 2. Sensitive Information
- **SSN:** XXX-XX-#### (Last 4: [####])
- **Credit Card:** ###-####-####-#### (Last 4: [####])

---

## 3. Issue Summary
[Concise 2–3 sentence description of the issue]

- **Amount in Question:** $XX.XX  
- **Date of Transaction:** MM/DD/YYYY

---

## 4. Resolution Details
[Step-by-step actions taken by agent]

- **Investigation Status:** [Initiated/Completed]  
- **Expected Resolution Time:** [If mentioned]

---

## 5. Outcome
- [ ] Resolved during call  
- [✔] Requires follow-up  
- **Next Steps:** [What customer should do]

---

## 6. Additional Insights
[Any relevant observations or recommendations]

---

<p align="right"><i>AI engine powered by IndominusLabs</i></p>
```

---

### **Style & Guidelines:**

- Use consistent bolding for labels (e.g., `**Label:**`).
- Use horizontal rules (`---`) to separate sections.
- Keep language formal, structured, and professional.
- Do **not** include any introductory phrases like "Here is the report".

---


    '''
)
# ---------------- Gemini Agent for cript---------------- #

model1 = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=settings.GROQ_API_KEY))
agent1 = Agent(model, system_prompt=f"""
           You are a dialogue formatting assistant. Your task is to process raw customer support transcripts that contain a mix of speech from a support agent and a client. These transcripts are not labeled by speaker.

Given a single input string of an entire conversation, your job is to:

Determine who is speaking at each point based on context, tone, and content.

Format the conversation in a clear, alternating format using the labels Support Agent: and Client: (with no asterisks, no bolding, and no Markdown).

Ensure each message is properly grouped so that a single person's multi-sentence response is kept together under one speaker label.

If the support agent introduces themselves or asks verification questions, it is a clear indicator of their voice.

If the person is describing a problem, sharing personal details, or expressing gratitude, that is usually the client.

Maintain all original wording and line breaks wherever possible.

Do not use Markdown formatting. Do not use asterisks under any circumstances.
    
    """)

# ---------------- Async Wrappers ---------------- #

async def get_transcript_from_groq(audio_path: str) -> str:
    client = Groq(api_key=settings.GROQ_API_KEY)
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    return response.text


async def transcribe_audio(audio_path: str):
    transcript = await get_transcript_from_groq(audio_path)
    sanitized = sanitize_transcript(transcript)

    # Get formatted speaker dialogue using sanitized transcript
    formatted_dialogue = await agent1.run(sanitized, model_settings={"temperature": 0.3})

    return sanitized, formatted_dialogue.data



async def generate_report(transcript: str):
    sanitized_transcript = sanitize_transcript(transcript)
    response = await agent.run(sanitized_transcript, model_settings={"temperature": 0.2})
    return response.data

async def transcript_conversion(sanitized_transcript: str):
    response = await agent.run(sanitized_transcript, model_settings={"temperature": 0.2})
    return response.data

# ---------------- PDF Utilities ---------------- #
def save_report_as_pdf(report_text):
    if not report_text.strip():
        return None

    pdf_buffer = BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)
    y_position = 750  

    for line in report_text.split("\n"):
        if y_position < 50:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 12)
            y_position = 750
        pdf_canvas.drawString(50, y_position, line)
        y_position -= 20

    pdf_canvas.save()
    return pdf_buffer.getvalue()

# ---------------- QA Agent ---------------- #
def create_qa_agent(report_data: str):
    model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=settings.GROQ_API_KEY))
    qa_agent = Agent(model, system_prompt=f"""
        You are a helpful assistant answering questions based on the structured report.
        Report: {report_data}
    """)
    return qa_agent

import streamlit as st
import asyncio
import base64

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="VoiceIQ", page_icon="🧠", layout="wide")

# Inline CSS styling
st.markdown("""
    <style>
    .app-title {
        font-size: 3em;
        font-weight: 800;
        color: #4a4a4a;
        margin-bottom: 0.2em;
    }
    .subheading {
        font-size: 1.2em;
        color: #6e6e6e;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)



# ---------------- App Title ---------------- #
st.markdown('<div class="app-title">VoiceIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="subheading">Your AI-Powered Customer Support Report Generator 🎧</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- Audio Upload & Transcription ---------------- #
st.header("🎤 Upload Audio for Transcription")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file and st.button("📝 Transcribe and Generate Report"):
    st.session_state.clear()
    with st.spinner("Transcribing..."):
        temp_audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        loop = asyncio.get_event_loop()
        sanitized, formatted_dialogue = loop.run_until_complete(transcribe_audio(temp_audio_path))
        st.session_state["sanitized"] = sanitized
        st.session_state["dialogue"] = formatted_dialogue

        st.text_area("🗣️ Detected Dialogue (Formatted):", formatted_dialogue, height=250)

    if st.session_state.get("sanitized"):
        with st.spinner("Generating report..."):
            report_result = loop.run_until_complete(generate_report(st.session_state["sanitized"]))
            st.session_state["report"] = report_result
            st.success("✅ Report Generated Successfully!")
            st.text_area("📑 Generated Report:", report_result, height=300)

st.markdown("---")

# ---------------- Manual Transcript Input ---------------- #
st.header("📝 Or Paste a Customer Support Transcript")

transcript_input = st.text_area("Paste the transcript below:")
if st.button("🚀 Generate Report"):
    sanitized_input = sanitize_transcript(transcript_input.strip())
    loop = asyncio.get_event_loop()
    formatted_dialogue = loop.run_until_complete(agent1.run(sanitized_input, model_settings={"temperature": 0.3}))

    st.session_state["sanitized"] = sanitized_input
    st.session_state["dialogue"] = formatted_dialogue.data
    st.text_area("🗣️ Detected Dialogue (Formatted):", formatted_dialogue.data, height=250)

    if sanitized_input:
        with st.spinner("Generating report..."):
            report_result = loop.run_until_complete(generate_report(sanitized_input))
            st.session_state["report"] = report_result
            st.success("✅ Report Generated Successfully!")
            st.text_area("📑 Generated Report:", report_result, height=300)
    else:
        st.error("❌ Please provide a transcript!")

st.markdown("---")

# ---------------- Report Download & Chatbot ---------------- #
if "report" in st.session_state and st.session_state["report"]:
    pdf_bytes = save_report_as_pdf(st.session_state["report"])
    if pdf_bytes:
        st.download_button("📥 Download Report as PDF", pdf_bytes, "customer_support_report.pdf", mime="application/pdf")

    st.header("📑 Preview Report")
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    st.markdown("---")
    st.header("💬 Chat with Your Report")

    qa_agent = create_qa_agent(st.session_state["report"])
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask something about the report...")
    if user_input:
        st.session_state.setdefault("messages", []).append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(qa_agent.run(user_input, model_settings={"temperature": 0.2}))
                bot_reply = response.data
                st.markdown(bot_reply)
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
