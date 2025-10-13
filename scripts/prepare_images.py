import os
import glob
import rawpy
import imageio
from tqdm import tqdm
import numpy as np
import shutil

# Imports for HEIC support
import pillow_heif
pillow_heif.register_heif_opener()

# Imports for Google Drive
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io


## --- CONFIGURATION --- ##
# Set your desired source here: 'local' or 'google_drive'
SOURCE_TYPE = 'local'

# --- Local Settings ---
# Use this for local folders, external drives, or exported Apple Photos
LOCAL_PATH = 'raw_images' 

# --- Google Drive Settings ---
# Find this in the URL of your Google Drive folder
GDRIVE_FOLDER_ID = 'YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE' 

# --- General Settings ---
OUTPUT_FOLDER = "HR_dataset"
# Add any other file extensions you want to process
SUPPORTED_EXTENSIONS = [
    ".arw", ".cr2", ".dng", # RAW files
    ".jpg", ".jpeg", ".png", # Standard images
    ".heic", ".heif"       # Apple format
]
## --------------------- ##


def authenticate_gdrive():
    """Handles Google Drive authentication."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.readonly'])
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', ['https://www.googleapis.com/auth/drive.readonly'])
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_gdrive_files(folder_id, output_dir):
    """Downloads files from a Google Drive folder and returns their local paths."""
    print("Connecting to Google Drive...")
    creds = authenticate_gdrive()
    service = build('drive', 'v3', credentials=creds)
    
    # Create a temporary download directory
    temp_download_dir = os.path.join(output_dir, "temp_gdrive_downloads")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    local_file_paths = []

    try:
        query = f"'{folder_id}' in parents"
        # results = service.files().list(q=query, pageSize=1000, fields="nextPageToken, files(id, name)").execute()
        # items = results.get('files', [])
        # New block with pagination
        items = []
        page_token = None
        while True:
            results = service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name)",
                pageToken=page_token
            ).execute()
            
            items.extend(results.get('files', []))
            page_token = results.get('nextPageToken', None)
            if page_token is None:
                break

        if not items:
            print("No files found in the specified Google Drive folder.")
            return []

        print(f"Found {len(items)} files in Google Drive. Starting download...")
        for item in tqdm(items, desc="Downloading from GDrive"):
            file_id = item['id']
            file_name = item['name']
            
            # Check if file type is supported before downloading
            if not any(file_name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue

            local_path = os.path.join(temp_download_dir, file_name)
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            local_file_paths.append(local_path)
            
        return local_file_paths

    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

def get_local_files(path):
    """Finds all supported image files in a local directory."""
    print(f"Scanning local directory: {path}")
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        # Search for both lowercase and uppercase extensions
        files.extend(glob.glob(os.path.join(path, f"**/*{ext.lower()}"), recursive=True))
        files.extend(glob.glob(os.path.join(path, f"**/*{ext.upper()}"), recursive=True))
    return list(set(files)) # Use set to remove duplicates

def process_files(file_list, output_folder):
    """Processes a list of image files (RAW, JPG, etc.) and saves them as PNGs."""
    if not file_list:
        print("No files to process.")
        return
        
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing {len(file_list)} files...")

    for file_path in tqdm(file_list, desc="Converting images"):
        try:
            base_name = os.path.basename(file_path)
            file_name, file_ext = os.path.splitext(base_name)
            output_path = os.path.join(output_folder, f"{file_name}.png")

            image_data = None
            if file_ext.lower() in ['.arw', '.cr2', '.dng']:
                # Process RAW files for maximum quality
                with rawpy.imread(file_path) as raw:
                    image_data = raw.postprocess(output_bps=16)
            else:
                # Process standard images like JPG, HEIC, PNG
                image_data = imageio.imread(file_path)
            
            # Save as a high-quality PNG
            imageio.imwrite(output_path, image_data)

        except Exception as e:
            print(f"\nCould not process {file_path}: {e}")

def main():
    """Main function to run the data preparation pipeline."""
    file_list = []
    if SOURCE_TYPE == 'local':
        file_list = get_local_files(LOCAL_PATH)
    elif SOURCE_TYPE == 'google_drive':
        file_list = get_gdrive_files(GDRIVE_FOLDER_ID, OUTPUT_FOLDER)
    else:
        print(f"Error: Unknown SOURCE_TYPE '{SOURCE_TYPE}'. Please choose 'local' or 'google_drive'.")
        return

    process_files(file_list, OUTPUT_FOLDER)
    
    # Add this cleanup block at the end
    if SOURCE_TYPE == 'google_drive':
        temp_download_dir = os.path.join(OUTPUT_FOLDER, "temp_gdrive_downloads")
        if os.path.exists(temp_download_dir):
            print("Cleaning up temporary download files...")
            shutil.rmtree(temp_download_dir)
            
    print(f"\n✅ All files have been processed and saved to the '{OUTPUT_FOLDER}' directory.")

if __name__ == '__main__':
    main()