# Newspeak House RAG Bot

A RAG (Retrieval-Augmented Generation) chatbot for Newspeak House, powered by Google Cloud VertexAI and Gemini 1.5.

## Overview

This bot provides intelligent responses to questions about Newspeak House by leveraging a corpus of documents stored in Google Drive. It uses:

- Google Cloud VertexAI for RAG implementation
- Gemini 1.5 for language model capabilities
- Google Cloud Storage for corpus metadata storage
- (Coming soon) Slack integration for user interaction

## Features

- Semantic search over Newspeak House documents
- Source attribution for all responses
- Efficient corpus management with caching
- Automatic updates when source documents change
- Detailed logging and error handling

## Prerequisites

- Python 3.9+
- Google Cloud Project with VertexAI API enabled
- Service account with necessary permissions
- Google Drive folder containing source documents
- (For deployment) Railway account

## Environment Variables

Create a `.env` file with:

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
DRIVE_FOLDER_ID=your-drive-folder-id
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/simonwisdom/nwspk-ragbot.git
cd nwspk-ragbot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the RAG pipeline:
```bash
python rag.py
```

The system will:
1. Initialize connection to Google Cloud
2. Set up or reuse the RAG corpus
3. Process queries using the Gemini 1.5 model
4. Provide responses with source citations

## Development

- `rag.py`: Main RAG pipeline implementation
- (Coming soon) `bot.py`: Slack bot integration
- (Coming soon) `api.py`: REST API for web interface

## Deployment

Instructions for Railway deployment coming soon.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Newspeak House for providing the source material
- Google Cloud for VertexAI and Gemini
- Railway for hosting 