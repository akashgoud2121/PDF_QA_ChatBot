# PDF_QA_ChatBot

# Financial Data Interactive QA Bot

## Overview
The Financial Data Interactive QA Bot allows users to upload financial documents, ask complex queries, and receive detailed, accurate answers using Generative AI. This application uses Streamlit for the user interface and integrates Google Generative AI capabilities for its backend.

## Features
- **PDF Upload**: Upload financial documents such as quarterly reports or balance sheets.
- **Question and Answer**: Ask targeted financial questions and get context-aware responses.
- **Gemini AI Integration**: Powered by Gemini AI models for advanced language understanding.

## Prerequisites
Ensure you have the following installed:

- Python 3.9 or newer
- Docker (for containerized deployment)
- A Gemini AI Studio API key

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo-name/financial-qa-bot.git
cd financial-qa-bot
```

### 2. Install Dependencies (For Local Setup)

Create a virtual environment and install required Python packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add Your Gemini API Key

1. Visit [Gemini AI Studio](https://studio.gemini.com) and log in.
2. Navigate to the **API Keys** section.
3. Click **Create New API Key**, name it, and copy the generated key.
4. Save the key as an environment variable:

   **Linux/macOS**:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   
   **Windows (Command Prompt)**:
   ```bash
   set GOOGLE_API_KEY="your_api_key_here"
   ```

### 4. Run the Application

Execute the application with:
```bash
streamlit run app.py
```
The app will be available at `http://localhost:8501`.

### 5. Using Docker for Deployment

#### Build the Docker Image
Ensure Docker is installed and running on your machine:
```bash
docker build -t financial-qa-bot .
```

#### Run the Docker Container
```bash
docker run -p 8501:8501 financial-qa-bot
```
Access the app at `http://localhost:8501`.

## Usage Instructions

### Step 1: Upload a Financial Document
1. Launch the app.
2. Use the **Upload a PDF** option on the sidebar to upload your file.

### Step 2: Ask a Financial Query
1. In the **Ask a Question** section, type your query.
2. Example queries:
   - "What are the total expenses for Q2 2023?"
   - "Show the operating margin for the past 6 months."

### Step 3: Interpret the Bot’s Response
The bot will return detailed, accurate answers based on the uploaded document. If the bot cannot find the answer, it will indicate that the information is not available.

## Example Interaction Scenarios

### Query 1: "What are the total expenses for Q2 2023?"
**Response**: "The total expenses for Q2 2023 are $250,000. This includes $150,000 in operational costs and $100,000 in marketing expenses."

### Query 2: "Show the operating margin for the past 6 months."
**Response**: "The operating margin for the past 6 months averages 15%, with a high of 18% in April 2023 and a low of 12% in January 2023."

## Deployment to Streamlit Community Cloud

To deploy the app on Streamlit Community Cloud:
1. Push your repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io).
3. Sign in with your GitHub account and select your repository.
4. Provide the `app.py` path for deployment.

## Troubleshooting

- **Error: `streamlit not found`**
  Ensure Streamlit is installed and included in your `requirements.txt`.
- **Docker build issues**
  Check the Dockerfile for completeness and ensure system dependencies are installed.

Developed by E Akash Goud 🌟.

