# Newspeak House RAG Bot

A RAG (Retrieval-Augmented Generation) chatbot for Newspeak House, powered by Google Cloud VertexAI and Gemini 1.5.

## Overview

This bot provides intelligent responses to questions about Newspeak House by leveraging a corpus of documents stored in Google Drive. It uses:

- Google Cloud VertexAI for RAG implementation
- Gemini 1.5 for language model capabilities
- Google Cloud Storage for corpus metadata storage
- Slack integration for user interaction

## Features

- Semantic search over Newspeak House documents
- Source attribution for all responses
- Efficient corpus management with caching
- Automatic updates when source documents change
- Detailed logging and error handling
- Thread-aware Slack bot integration
- Context-aware responses

## Prerequisites

- Python 3.9+
- Google Cloud Project with VertexAI API enabled
- Service account with necessary permissions
- Google Drive folder containing source documents
- Slack workspace with admin access
- (For deployment) Railway account

## Slack Setup

1. Create a new Slack App at https://api.slack.com/apps
2. Configure Bot User:
   - Add a Bot User named "Ragbot"
   - Set "Always Show My Bot as Online" to On

3. Set up OAuth & Permissions:
   - Add these Bot Token Scopes:
     ```
     app_mentions:read
     channels:history
     channels:read
     chat:write
     groups:history
     groups:read
     im:history
     im:read
     reactions:read
     team:read
     threads:read
     users:read
     ```

4. Enable Event Subscriptions:
   - Subscribe to bot events:
     ```
     app_mention
     message.channels
     message.groups
     message.im
     ```

5. Install App to Workspace:
   - Get Bot User OAuth Token
   - Get Signing Secret
   - Get App Token

## Environment Variables

Create a `.env` file with:

```bash
# Google Cloud Settings
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
DRIVE_FOLDER_ID=your-drive-folder-id

# Slack Settings
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_APP_TOKEN=xapp-your-app-token

# Server Settings (for Railway)
PORT=3000
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

1. Start the RAG pipeline and Slack bot:
```bash
python bot.py
```

2. In Slack:
   - Add the bot to your desired channels
   - Mention the bot with a question: "@Ragbot what is Newspeak House?"
   - The bot will respond with information from the document corpus
   - Use threads for follow-up questions to maintain context

## Development

- `rag.py`: Main RAG pipeline implementation
- `bot.py`: Slack bot integration
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