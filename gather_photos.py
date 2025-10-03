import os
import io
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List

# (Copy the Google auth functions from our previous script)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
# A Note on Google Photos: 
# The best way to get original-quality files is to use Google Takeout to export your albums directly to a folder in Google Drive.
# Then, use this script to download that folder.

## --- CONFIGURATION --- ##
# What source are you gathering from today? 'gdrive' , 'gphotos'
SOURCE_TO_GATHER = 'gdrive'

# --- Google Drive Settings ---
GDRIVE_FOLDER_ID = 'YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE'
GDRIVE_DESTINATION = Path.home() / "Desktop" / "Photos" / "GDrive"

# (You can add another section here for your GPhotos Takeout folder)
# GPHOTOS_FOLDER_ID = 'YOUR_GOOGLE_PHOTOS_TAKEOUT_FOLDER_ID'
# GPHOTOS_DESTINATION = Path.home() / "Desktop" / "Photos" / "GPhotos"
## --------------------- ##

# (Paste the exact authenticate_gdrive() function from our previous script here)
def authenticate_gdrive() -> Credentials:
    # ... (same function as before) ...
    creds = None
    token_file = Path('token.json')
    creds_file = Path('credentials.json')
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), ['https://www.googleapis.com/auth/drive.readonly'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), ['https://www.googleapis.com/auth/drive.readonly'])
            creds = flow.run_local_server(port=0)
        with token_file.open('w') as token:
            token.write(creds.to_json())
    return creds

def download_gdrive_folder(folder_id: str, destination_path: Path):
    """Downloads all files from a Google Drive folder to a local destination."""
    destination_path.mkdir(exist_ok=True, parents=True)
    # (This is a slightly modified get_gdrive_files function)
    print(f"Connecting to Google Drive to download folder '{folder_id}'...")
    creds = authenticate_gdrive()
    service = build('drive', 'v3', credentials=creds)
    
    try:
        items = []
        page_token = None
        while True:
            results = service.files().list(q=f"'{folder_id}' in parents", pageSize=1000, fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if page_token is None: break

        if not items:
            print("No files found.")
            return

        print(f"Found {len(items)} files. Downloading to '{destination_path}'...")
        for item in tqdm(items, desc="Downloading files"):
            local_path = destination_path / item['name']
            request = service.files().get_media(fileId=item['id'])
            with local_path.open('wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
        print("✅ GDrive download complete.")

    except HttpError as error:
        print(f'An error occurred: {error}')

def main():
    """Main function to run the data gathering pipeline."""
    if SOURCE_TO_GATHER == 'gdrive':
        download_gdrive_folder(GDRIVE_FOLDER_ID, GDRIVE_DESTINATION)
    # elif SOURCE_TO_GATHER == 'gphotos':
    #     download_gdrive_folder(GPHOTOS_FOLDER_ID, GPHOTOS_DESTINATION)
    else:
        print(f"Error: Unknown source '{SOURCE_TO_GATHER}'.")

if __name__ == '__main__':
    main()