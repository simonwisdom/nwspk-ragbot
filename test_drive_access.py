from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up credentials
credentials = service_account.Credentials.from_service_account_file(
    os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)

# Build the Drive API service
service = build('drive', 'v3', credentials=credentials)

# Folder ID from your config
FOLDER_ID = "1U8TEZKLv3h1F-x9CkNFfG8vimojyYikn"

try:
    # List files in the folder
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents",
        pageSize=10,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    
    items = results.get('files', [])
    
    if not items:
        print('No files found in the folder.')
    else:
        print('Files in the folder:')
        for item in items:
            print(f"{item['name']} ({item['id']}) - {item['mimeType']}")
            
except Exception as e:
    print(f"Error accessing Drive: {str(e)}") 