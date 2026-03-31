# 📝 AI Text Summarizer

<div align="center">

![AI Text Summarizer](https://capsule-render.vercel.app/api?type=waving&color=0a0a0f,7fffb2,3af0ff&height=200&section=header&text=AI%20Text%20Summarizer&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Intelligent%20summarization%20powered%20by%20T5&descAlignY=58&descSize=16)

<br/>

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-00C7B7?style=for-the-badge)](https://ai-text-summarizer-001.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-7fffb2?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=statuspage&logoColor=white)]()
[![Model](https://img.shields.io/badge/Model-T5--Small-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/t5-small)

<br/>

> **Paste any text. Get a crisp 2–3 line summary. Instantly.**  
> Built with T5-Small, HuggingFace Transformers, and Streamlit — deployed on the cloud.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-00C7B7?style=for-the-badge)](https://ai-text-summarizer-001.streamlit.app)
</div>

---

## 🖼️ Preview

<div align="center">

```
┌───────────────────────────────────────────────┐
│                                               │
│   ✦ Powered by T5 · Transformers             │
│                                               │
│   AI Text                                     │
│   Summarizer                                  │
│   ──────                                      │
│                                               │
│   [ Paste your text here...           ]       │
│                                  0 chars      │
│                                               │
│   ⚡ Generate Summary                        │
│                                               │
│   ✦ SUMMARY                                  │
│   Your intelligent summary appears here...   │
│                                              │
└──────────────────────────────────────────────┘
```

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Summarization** | Uses Google's T5-Small model for intelligent text summarization |
| ⚡ **Instant Results** | Generates summaries in seconds on CPU |
| 🎨 **Modern Dark UI** | Custom dark theme with gradient accents and smooth animations |
| 📊 **Live Char Counter** | Real-time character count as you type |
| 🔒 **Model Caching** | `@st.cache_resource` ensures model loads only once |
| ☁️ **Cloud Deployed** | Live on Streamlit Cloud — no setup needed |
| 📱 **Responsive** | Works on desktop and mobile browsers |

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS |
| **ML Model** | T5-Small (HuggingFace) |
| **Framework** | HuggingFace Transformers |
| **Deep Learning** | PyTorch (CPU mode) |
| **Hosting** | Streamlit Cloud |
| **Language** | Python 3.14 |

</div>

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed:

```bash
python --version  # 3.9+ recommended
```

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/asad-aziz-001/ai-text-summarizer.git
cd ai-text-summarizer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```txt
streamlit
transformers>=4.45.0,<5.0.0
torch>=2.9.0
sentencepiece
```

---

## 📁 Project Structure

```
ai-text-summarizer/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 💡 How It Works

```
User Input Text
      │
      ▼
┌─────────────┐
│  Streamlit  │  ← Web Interface
│     UI      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   T5-Small  │  ← HuggingFace Pipeline
│    Model    │     (Cached on first load)
└──────┬──────┘
       │
       ▼
  Summary Output
  (2–3 sentences)
```

1. User pastes text into the input area
2. Clicks **⚡ Generate Summary**
3. Text is passed to the `summarization` pipeline
4. T5-Small model processes and compresses the text
5. Clean summary is displayed in the result card

---

## 🎨 UI Highlights

- **Color Palette** — Deep navy (`#0a0a0f`) with neon green (`#7fffb2`) and cyan (`#3af0ff`) accents
- **Typography** — `Syne` for headings, `DM Sans` for body text
- **Gradient Title** — White → Green → Cyan animated gradient
- **Glow Effects** — Subtle radial gradients on background
- **Animated Button** — Hover lift with shadow bloom
- **Result Card** — Gradient top border with glassmorphism background

---

## ⚙️ Configuration

You can tweak summarization behavior in `app.py`:

```python
summary = summarizer(
    input_text,
    max_length=60,   # Maximum summary length (tokens)
    min_length=20,   # Minimum summary length (tokens)
    do_sample=False  # Deterministic output
)
```

| Parameter | Default | Description |
|---|---|---|
| `max_length` | `60` | Max tokens in summary |
| `min_length` | `20` | Min tokens in summary |
| `do_sample` | `False` | Greedy decoding (consistent results) |
| `device` | `-1` | `-1` = CPU, `0` = GPU |

---

## 🧪 Test Inputs

Try these sample texts to test the app:

<details>
<summary><b>🤖 Technology</b></summary>

```
Artificial intelligence is rapidly transforming industries across the globe. From healthcare to finance, AI-powered systems are automating complex tasks, improving accuracy, and reducing costs. Machine learning algorithms can now detect diseases from medical images with higher accuracy than human doctors. In finance, AI is used for fraud detection, algorithmic trading, and personalized banking. However, concerns about job displacement, data privacy, and algorithmic bias continue to grow as AI becomes more integrated into daily life.
```
</details>

<details>
<summary><b>🌍 Climate Change</b></summary>

```
Climate change is one of the most pressing challenges facing humanity today. Rising global temperatures caused by greenhouse gas emissions are leading to more frequent and intense natural disasters, including hurricanes, floods, and wildfires. Sea levels are rising, threatening coastal cities around the world. Scientists warn that without immediate and drastic action to reduce carbon emissions, the consequences could be irreversible.
```
</details>

<details>
<summary><b>🇵🇰 Pakistan Tech</b></summary>

```
Pakistan is one of the world's fastest-growing digital economies. With over 100 million internet users and a young population, the country has seen a boom in e-commerce, fintech, and freelancing. Pakistani freelancers rank among the top earners globally on platforms like Upwork and Fiverr. The government has launched several digital initiatives to improve connectivity in rural areas and promote tech startups.
```
</details>

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs via [Issues](https://github.com/asad-aziz-001/ai-text-summarizer/issues)
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">

**Made with ❤️ by [Asad Aziz](https://asad-aziz-001.github.io/Portfolio/)**

[![GitHub](https://img.shields.io/badge/GitHub-asad--aziz--001-181717?style=flat-square&logo=github)](https://github.com/asad-aziz-001)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Asad%20Aziz-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/asad-aziz-ai)

[![Portfolio](https://img.shields.io/badge/Portfolio-asadaziz.dev-7fffb2?style=flat-square&logo=vercel&logoColor=black)](https://asad-aziz-001.github.io/Portfolio/)


*If this helped you, please ⭐ star the repo!*

</div>
