# app.py - Fixed version that properly mirrors your working scripts
from flask import Flask, jsonify, session, request, redirect
from flask_cors import CORS
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import json
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import base64
from bs4 import BeautifulSoup
import html
import time
import re
from flask import Flask, send_from_directory
import os
import os
from flask import Flask, jsonify, session, request, redirect, send_from_directory

# Import transformers for models
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline
)
import torch

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-for-job-tracker')

# CORS configuration
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000"],
     supports_credentials=True)

# OAuth Configuration
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
BACKEND_PORT = os.getenv('BACKEND_PORT', '5001')

# Create data directory if it doesn't exist
DATA_DIR = 'user_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Global variables for models (loaded once at startup)
classifier_model = None
ner_model = None
ner_tokenizer = None
ner_pipeline = None


#here
def load_models():
    """Load AI models from Hugging Face Hub."""
    global classifier_model, ner_model, ner_tokenizer, ner_pipeline
    
    try:
        # Load classifier directly from Hugging Face Hub
        print("🌐 Loading email classifier from Hugging Face...")
        classifier_model = pipeline(
            "text-classification",
            model="Minaides/job-email-classifier",  # Hugging Face repo
            device=-1  # CPU
        )
        print("✅ Classifier loaded successfully from Hugging Face!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load classifier: {e}")
        classifier_model = None
    
    try:
        # Load NER model directly from Hugging Face Hub
        print("🌐 Loading NER model from Hugging Face...")
        model_path = "Minaides/job_ner_model"
        
        ner_tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)
        ner_model = AutoModelForTokenClassification.from_pretrained(model_path, use_auth_token=True)
        
        ner_pipeline = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_tokenizer,
            aggregation_strategy="simple",
            device=-1  # CPU
        )
        print("✅ NER model loaded successfully from Hugging Face!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load NER model: {e}")
        ner_pipeline = None

#to here

# Load models at startup
load_models()

def get_client_config():
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [f"http://localhost:{BACKEND_PORT}/auth/callback"]
        }
    }

def get_user_data_file():
    """Get file path for user's data based on session ID"""
    user_id = session.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
    return os.path.join(DATA_DIR, f'{user_id}_results.json')

def clean_email_body(text):
    """Clean and normalize email body text - from job-fetcher.py"""
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Fix common mojibake issues
    text = text.replace('â€™', "'")
    text = text.replace('â€"', "—")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    return text

def fetch_email(service, msg_id, max_retries=4):
    """Fetch email metadata + body - from job-fetcher.py"""
    attempt = 0
    wait = 1
    while attempt < max_retries:
        try:
            m = service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            headers = {h['name']: h['value'] for h in m.get('payload', {}).get('headers', [])}

            def extract_body_text(part):
                """Recursively extract all text from email parts."""
                texts = []

                if part.get('parts'):
                    for subpart in part['parts']:
                        texts.append(extract_body_text(subpart))

                body = part.get('body', {})
                data = body.get('data')

                if data:
                    try:
                        decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        mime_type = part.get('mimeType', '')
                        
                        if mime_type == 'text/html':
                            soup = BeautifulSoup(decoded, 'html.parser')
                            decoded = soup.get_text(separator="\n")
                        
                        texts.append(clean_email_body(decoded))
                    except Exception as e:
                        print(f"Error decoding part: {e}")

                return "\n".join([t for t in texts if t])

            body = extract_body_text(m.get('payload', {}))

            return {
                'gmail_id': m.get('id'),
                'thread_id': m.get('threadId'),
                'subject': headers.get('Subject', ''),
                'from': headers.get('From', ''),
                'date': headers.get('Date', ''),
                'snippet': m.get('snippet', ''),
                'internal_date': m.get('internalDate', ''),
                'body': body
            }

        except HttpError as e:
            attempt += 1
            print(f"Error fetching email (attempt {attempt}/{max_retries}): {e}")
            time.sleep(wait)
            wait *= 2

    return None

def classify_emails(emails_df):
    """Classify emails - exactly like test_classifier_on_emails.py"""
    if classifier_model is None:
        print("⚠️ Classifier not loaded, marking all as job applications")
        emails_df['is_application'] = True
        emails_df['confidence'] = 0.5
        return emails_df
    
    print(f"📧 Classifying {len(emails_df)} emails...")
    
    results = []
    for idx, row in emails_df.iterrows():
        email_text = f"Subject: {row['subject']}\n\n{row['body'][:1000]}"
        
        try:
            classification = classifier_model(email_text)[0]
            
            results.append({
                'gmail_id': row['gmail_id'],
                'is_application': classification['label'] == 'JOB_APPLICATION',
                'confidence': classification['score']
            })
            
            if idx % 10 == 0:
                print(f"Classified {idx}/{len(emails_df)}...")
                
        except Exception as e:
            print(f"Error classifying email: {e}")
            results.append({
                'gmail_id': row['gmail_id'],
                'is_application': False,
                'confidence': 0.0
            })
    
    # Merge classification results with original dataframe
    results_df = pd.DataFrame(results)
    emails_df = emails_df.merge(results_df, on='gmail_id', how='left')
    
    application_count = len(emails_df[emails_df['is_application'] == True])
    print(f"✅ Found {application_count} job applications out of {len(emails_df)} emails")
    
    return emails_df

def extract_job_info(emails_df):
    """Extract company and position - exactly like 5_inference.py"""
    if ner_pipeline is None:
        print("⚠️ NER model not loaded, skipping extraction")
        emails_df['company'] = None
        emails_df['position'] = None
        return emails_df
    
    print(f"🔍 Extracting job information from {len(emails_df)} emails...")
    
    results = []
    for idx, row in emails_df.iterrows():
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(emails_df)} emails...")
        
        # Combine subject and body for context
        full_text = f"Subject: {row['subject']}\n\n{row['body'][:2000]}"
        
        try:
            # Run NER
            entities = ner_pipeline(full_text)
            
            # Group entities by type
            companies = []
            positions = []
            
            for entity in entities:
                if entity['entity_group'] == 'COMPANY':
                    companies.append(entity['word'].strip())
                elif entity['entity_group'] == 'POSITION':
                    positions.append(entity['word'].strip())
            
            # Clean and deduplicate
            companies = list(dict.fromkeys([c for c in companies if len(c) > 2]))
            positions = list(dict.fromkeys([p for p in positions if len(p) > 2]))
            
            results.append({
                'gmail_id': row['gmail_id'],
                'company': companies[0] if companies else None,
                'position': positions[0] if positions else None,
                'all_companies': companies,
                'all_positions': positions
            })
            
        except Exception as e:
            print(f"Error extracting from email: {e}")
            results.append({
                'gmail_id': row['gmail_id'],
                'company': None,
                'position': None,
                'all_companies': [],
                'all_positions': []
            })
    
    # Merge extraction results with original dataframe
    results_df = pd.DataFrame(results)
    emails_df = emails_df.merge(results_df, on='gmail_id', how='left')
    
    companies_found = len(emails_df[emails_df['company'].notna()])
    positions_found = len(emails_df[emails_df['position'].notna()])
    print(f"📊 Found {companies_found} companies and {positions_found} positions")
    
    return emails_df
if os.getenv('RENDER'):
    ALLOWED_ORIGINS = [
        "https://your-app-name.onrender.com",
        "http://localhost:3000"
    ]
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

CORS(app, 
     origins=ALLOWED_ORIGINS,
     supports_credentials=True)
@app.route('/')
def home():
    return jsonify({"message": "Job Application Tracker API - Running!"})

@app.route('/api/test')
def test_api():
    return jsonify({'message': 'API is working!'})
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    else:
        return send_from_directory('build', 'index.html')
@app.route('/auth/url')
def get_auth_url():
    try:
        flow = Flow.from_client_config(
            get_client_config(),
            scopes=SCOPES,
            redirect_uri=f'http://localhost:{BACKEND_PORT}/auth/callback'
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='false',
            prompt='consent'
        )
        
        session['state'] = state
        return jsonify({'auth_url': authorization_url})
        
    except Exception as e:
        return jsonify({'error': f'Failed to create auth URL: {str(e)}'}), 500

@app.route('/auth/callback')
def auth_callback():
    try:
        state = session.get('state')
        
        if not state:
            return redirect(f'{FRONTEND_URL}?auth=error&message=State missing')
            
        flow = Flow.from_client_config(
            get_client_config(),
            scopes=SCOPES,
            state=state,
            redirect_uri=f'http://localhost:{BACKEND_PORT}/auth/callback'
        )
        
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        
        # Store credentials in session
        session['credentials'] = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        session['authenticated'] = True
        
        return redirect(f'{FRONTEND_URL}?auth=success')
        
    except Exception as e:
        return redirect(f'{FRONTEND_URL}?auth=error&message={str(e)}')
# Add this route to serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    else:
        return send_from_directory('build', 'index.html')
@app.route('/auth/status')
def auth_status():
    # Check if we have results file
    has_results = os.path.exists(get_user_data_file())
    return jsonify({
        'authenticated': session.get('authenticated', False),
        'email_processing': session.get('email_processing', False),
        'has_results': has_results,
        'models_loaded': {
            'classifier': classifier_model is not None,
            'ner': ner_pipeline is not None
        }
    })

@app.route('/api/process-emails')
def process_emails():
    if 'credentials' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        session['email_processing'] = True
        
        # Build Gmail service
        credentials = Credentials(**session['credentials'])
        service = build('gmail', 'v1', credentials=credentials)
        
        # Build date range (one year ago -> today) - exactly like job-fetcher.py
        today = datetime.today().date()
        one_year_ago = (today - timedelta(days=365)).strftime("%Y/%m/%d")
        tomorrow = (today + timedelta(days=1)).strftime("%Y/%m/%d")
        
        query = f"after:{one_year_ago} before:{tomorrow} subject:(application OR applying)"
        
        print(f"🔍 Searching emails with query: {query}")
        
        # Fetch message IDs
        all_messages = []
        page_token = None
        
        while True:
            try:
                results = service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=500,
                    pageToken=page_token
                ).execute()
                
                msgs = results.get('messages', [])
                all_messages.extend(msgs)
                page_token = results.get('nextPageToken')
                
                if not page_token:
                    break
                    
            except HttpError as e:
                print(f"Error fetching message IDs: {e}")
                break
        
        print(f"📧 Found {len(all_messages)} emails to process")
        
        if len(all_messages) == 0:
            session['email_processing'] = False
            return jsonify({'error': 'No emails found with "application" or "applying" in subject'}), 404
        
        # Fetch full email data
        emails = []
        for i, msg in enumerate(all_messages):
            if i % 10 == 0:
                print(f"Fetching email {i+1}/{len(all_messages)}")
            
            email_data = fetch_email(service, msg['id'])
            if email_data:
                emails.append(email_data)
        
        # Convert to DataFrame
        emails_df = pd.DataFrame(emails)
        print(f"✅ Successfully fetched {len(emails_df)} emails")
        
        # Step 1: Classify emails (like test_classifier_on_emails.py)
        emails_df = classify_emails(emails_df)
        
        # Filter to only job applications
        job_applications = emails_df[emails_df['is_application'] == True].copy()
        print(f"🎯 {len(job_applications)} emails classified as job applications")
        
        # Step 2: Extract job info from applications only (like 5_inference.py)
        if len(job_applications) > 0:
            job_applications = extract_job_info(job_applications)
        
        # Save all results
        results_data = {
            'all_emails': emails_df.to_dict('records'),
            'job_applications': job_applications.to_dict('records'),
            'timestamp': datetime.now().isoformat(),
            'stats': {
                'total_emails': len(emails_df),
                'job_applications': len(job_applications),
                'companies_found': len(job_applications[job_applications['company'].notna()]),
                'positions_found': len(job_applications[job_applications['position'].notna()])
            }
        }
        
        # Save to file
        file_path = get_user_data_file()
        with open(file_path, 'w') as f:
            json.dump(results_data, f)
        
        session['email_processing'] = False
        
        return jsonify({
            'status': 'success',
            'processed': len(emails_df),
            'applications': len(job_applications),
            'companies_found': results_data['stats']['companies_found'],
            'positions_found': results_data['stats']['positions_found'],
            'message': f'Found {len(job_applications)} job applications in {len(emails_df)} emails'
        })
        
    except Exception as e:
        session['email_processing'] = False
        print(f"❌ ERROR in process_emails: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/results')
def get_results():
    try:
        file_path = get_user_data_file()
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'No results available. Please process emails first.'}), 404
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Return the job applications and stats
        return jsonify({
            'job_applications': data.get('job_applications', []),
            'stats': data.get('stats', {}),
            'timestamp': data.get('timestamp', '')
        })
        
    except Exception as e:
        print(f"❌ Error in get_results: {e}")
        return jsonify({'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/api/clear')
def clear_session():
    """Clear session and data files"""
    try:
        # Delete user data file
        file_path = get_user_data_file()
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Clear session
        session.clear()
        return jsonify({'message': 'Session and data cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
    print(f"🚀 Starting Job Tracker API on port {BACKEND_PORT}")
    app.run(debug=True, port=int(BACKEND_PORT), host='0.0.0.0')
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    if os.getenv('RENDER'):
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=True, port=port, host='0.0.0.0')