# app.py - Uses Hugging Face Inference API (no local model loading)
from flask import Flask, jsonify, session, request, redirect, send_from_directory
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
import requests  # For HF API calls
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='frontend/build', static_url_path='/')
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-for-job-tracker')

# Update CORS for production
if os.getenv('RENDER'):
    ALLOWED_ORIGINS = [
        "https://job-tracker-app-gv0b.onrender.com",
        "http://localhost:3000"
    ]
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

CORS(app, 
     origins=ALLOWED_ORIGINS,
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

# Hugging Face API Configuration
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# API URLs - Using free public models
CLASSIFIER_API = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
NER_API = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
QA_API = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

def load_models():
    """Initialize - no actual model loading, just verify API access"""
    print("✅ Using Hugging Face Inference API - no local models loaded")
    print(f"📡 Classifier API: {CLASSIFIER_API}")
    print(f"📡 NER API: {NER_API}")
    print(f"📡 QA API: {QA_API}")
    
    # Test API connectivity
    try:
        test_response = requests.post(
            CLASSIFIER_API, 
            headers=HEADERS, 
            json={
                "inputs": "test",
                "parameters": {"candidate_labels": ["job application", "other"]}
            }, 
            timeout=10
        )
        if test_response.status_code == 503:
            print("⚠️ Models are loading on HF servers, first calls may be slower")
        elif test_response.status_code == 200:
            print("✅ Classifier API is ready")
        else:
            print(f"⚠️ Classifier API status: {test_response.status_code}")
    except Exception as e:
        print(f"⚠️ Could not reach HF API: {e}")

# Initialize on startup
load_models()

def get_client_config():
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [
                "https://job-tracker-app-gv0b.onrender.com/auth/callback",
                f"http://localhost:{BACKEND_PORT}/auth/callback"
            ]
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
    """Clean and normalize email body text"""
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
    """Fetch email metadata + body"""
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

def classify_single_email(row, job_statuses):
    """Classify a single email - helper for parallel processing"""
    email_text = f"Subject: {row['subject']}\n\n{row['body'][:1000]}"
    
    try:
        response = requests.post(
            CLASSIFIER_API, 
            headers=HEADERS,
            json={
                "inputs": email_text,
                "parameters": {"candidate_labels": job_statuses}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            top_label = data.get('labels', [''])[0]
            confidence = data.get('scores', [0.0])[0]
            is_app = top_label in ["application submitted", "job interview invitation", "job rejection"]
            
            return {
                'gmail_id': row['gmail_id'],
                'is_application': is_app,
                'application_status': top_label if is_app else None,
                'confidence': float(confidence)
            }
        elif response.status_code == 503:
            time.sleep(20)
            retry_response = requests.post(
                CLASSIFIER_API, 
                headers=HEADERS,
                json={
                    "inputs": email_text,
                    "parameters": {"candidate_labels": job_statuses}
                },
                timeout=30
            )
            if retry_response.status_code == 200:
                data = retry_response.json()
                top_label = data.get('labels', [''])[0]
                confidence = data.get('scores', [0.0])[0]
                is_app = top_label in ["application submitted", "job interview invitation", "job rejection"]
                
                return {
                    'gmail_id': row['gmail_id'],
                    'is_application': is_app,
                    'application_status': top_label if is_app else None,
                    'confidence': float(confidence)
                }
        
        return {
            'gmail_id': row['gmail_id'],
            'is_application': False,
            'application_status': None,
            'confidence': 0.0
        }
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {
            'gmail_id': row['gmail_id'],
            'is_application': False,
            'application_status': None,
            'confidence': 0.0
        }

def classify_emails(emails_df):
    """Classify emails using HF Zero-Shot Classification API with parallel processing"""
    print(f"📧 Classifying {len(emails_df)} emails using HF API (parallel)...")
    
    # Define the 4 status types
    JOB_STATUSES = ["application submitted", "job interview invitation", "job rejection", "other"]
    
    results = []
    
    # Use ThreadPoolExecutor for parallel API calls (max 5 at once to avoid rate limits)
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(classify_single_email, row, JOB_STATUSES): idx 
            for idx, row in emails_df.iterrows()
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 10 == 0:
                print(f"Classified {completed}/{len(emails_df)}...")
    
    # Merge classification results with original dataframe
    results_df = pd.DataFrame(results)
    emails_df = emails_df.merge(results_df, on='gmail_id', how='left')
    
    application_count = len(emails_df[emails_df['is_application'] == True])
    
    # Count by status
    status_counts = emails_df[emails_df['is_application'] == True]['application_status'].value_counts()
    print(f"✅ Found {application_count} job applications out of {len(emails_df)} emails")
    print(f"   📝 Submitted: {status_counts.get('application submitted', 0)}")
    print(f"   📞 Interviews: {status_counts.get('job interview invitation', 0)}")
    print(f"   ❌ Rejections: {status_counts.get('job rejection', 0)}")
    
    return emails_df

def extract_single_job_info(row):
    """Extract job info from a single email - helper for parallel processing"""
    full_text = f"Subject: {row['subject']}\n\n{row['body'][:2000]}"
    
    companies = []
    position = None
    
    try:
        # Step 1: Extract companies using NER
        ner_response = requests.post(
            NER_API,
            headers=HEADERS,
            json={"inputs": full_text},
            timeout=60
        )
        
        if ner_response.status_code == 200:
            entities = ner_response.json()
            for entity in entities:
                entity_group = entity.get('entity_group', entity.get('entity', ''))
                word = entity.get('word', '').strip()
                if 'ORG' in entity_group and len(word) > 2:
                    word = word.replace('##', '')
                    companies.append(word)
            companies = list(dict.fromkeys([c for c in companies if len(c) > 2]))
        elif ner_response.status_code == 503:
            time.sleep(15)
            retry_ner = requests.post(NER_API, headers=HEADERS, json={"inputs": full_text}, timeout=60)
            if retry_ner.status_code == 200:
                entities = retry_ner.json()
                for entity in entities:
                    entity_group = entity.get('entity_group', entity.get('entity', ''))
                    word = entity.get('word', '').strip()
                    if 'ORG' in entity_group and len(word) > 2:
                        word = word.replace('##', '')
                        companies.append(word)
                companies = list(dict.fromkeys([c for c in companies if len(c) > 2]))
        
        # Step 2: Extract position using Question-Answering
        qa_response = requests.post(
            QA_API,
            headers=HEADERS,
            json={
                "inputs": {
                    "question": "What is the exact job title or position name that was applied for?",
                    "context": full_text
                }
            },
            timeout=60
        )
        
        if qa_response.status_code == 200:
            qa_data = qa_response.json()
            if qa_data.get('score', 0) > 0.1:
                position = qa_data.get('answer', '').strip()
                if len(position) < 5 or len(position) > 100:
                    position = None
        elif qa_response.status_code == 503:
            time.sleep(15)
            retry_qa = requests.post(
                QA_API,
                headers=HEADERS,
                json={
                    "inputs": {
                        "question": "What is the exact job title or position name that was applied for?",
                        "context": full_text
                    }
                },
                timeout=60
            )
            if retry_qa.status_code == 200:
                qa_data = retry_qa.json()
                if qa_data.get('score', 0) > 0.1:
                    position = qa_data.get('answer', '').strip()
                    if len(position) < 5 or len(position) > 100:
                        position = None
        
        return {
            'gmail_id': row['gmail_id'],
            'company': companies[0] if companies else None,
            'position': position,
            'all_companies': companies,
        }
    except Exception as e:
        print(f"Error extracting from email: {e}")
        return {
            'gmail_id': row['gmail_id'],
            'company': None,
            'position': None,
            'all_companies': [],
        }

def extract_job_info(emails_df):
    """Extract job info using HF NER API for companies and QA API for positions with parallel processing"""
    print(f"🔍 Extracting job information from {len(emails_df)} emails using HF API (parallel)...")
    
    results = []
    
    # Use ThreadPoolExecutor for parallel API calls (max 5 at once)
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(extract_single_job_info, row): idx 
            for idx, row in emails_df.iterrows()
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_row):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 10 == 0:
                print(f"Processed {completed}/{len(emails_df)} emails...")
    
    # Merge extraction results with original dataframe
    results_df = pd.DataFrame(results)
    emails_df = emails_df.merge(results_df, on='gmail_id', how='left')
    
    companies_found = len(emails_df[emails_df['company'].notna()])
    positions_found = len(emails_df[emails_df['position'].notna()])
    print(f"📊 Found {companies_found} companies and {positions_found} positions")
    
    return emails_df

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # Skip API routes
    if path.startswith('api/') or path.startswith('auth/'):
        return jsonify({'error': 'Not found'}), 404
    
    # Use the static_folder we configured
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        if os.path.exists(os.path.join(app.static_folder, 'index.html')):
            return send_from_directory(app.static_folder, 'index.html')
        else:
            # If no build folder, return API status
            return jsonify({"message": "Job Application Tracker API - Running!"})

@app.route('/api/test')
def test_api():
    return jsonify({'message': 'API is working!'})

@app.route('/auth/url')
def get_auth_url():
    try:
        # Use production URL if on Render
        redirect_uri = f'http://localhost:{BACKEND_PORT}/auth/callback'
        if os.getenv('RENDER'):
            redirect_uri = 'https://job-tracker-app-gv0b.onrender.com/auth/callback'
        
        flow = Flow.from_client_config(
            get_client_config(),
            scopes=SCOPES,
            redirect_uri=redirect_uri
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
        
        redirect_uri = f'http://localhost:{BACKEND_PORT}/auth/callback'
        if os.getenv('RENDER'):
            redirect_uri = 'https://job-tracker-app-gv0b.onrender.com/auth/callback'
            
        flow = Flow.from_client_config(
            get_client_config(),
            scopes=SCOPES,
            state=state,
            redirect_uri=redirect_uri
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

@app.route('/auth/status')
def auth_status():
    # Check if we have results file
    has_results = os.path.exists(get_user_data_file())
    return jsonify({
        'authenticated': session.get('authenticated', False),
        'email_processing': session.get('email_processing', False),
        'has_results': has_results,
        'models_loaded': {
            'classifier': True,  # Always true with API
            'ner': True  # Always true with API
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
        
        # Build date range (one year ago -> today)
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
        
        # Step 1: Classify emails
        emails_df = classify_emails(emails_df)
        
        # Filter to only job applications
        job_applications = emails_df[emails_df['is_application'] == True].copy()
        print(f"🎯 {len(job_applications)} emails classified as job applications")
        
        # Step 2: Extract job info from applications only
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
                'companies_found': len(job_applications[job_applications['company'].notna()]) if len(job_applications) > 0 else 0,
                'positions_found': len(job_applications[job_applications['position'].notna()]) if len(job_applications) > 0 else 0,
                'status_breakdown': {
                    'submitted': int(len(job_applications[job_applications['application_status'] == 'application submitted'])) if len(job_applications) > 0 else 0,
                    'interviews': int(len(job_applications[job_applications['application_status'] == 'job interview invitation'])) if len(job_applications) > 0 else 0,
                    'rejections': int(len(job_applications[job_applications['application_status'] == 'job rejection'])) if len(job_applications) > 0 else 0
                }
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
            'status_breakdown': results_data['stats']['status_breakdown'],
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
    port = int(os.getenv('PORT', 5001))
    if os.getenv('RENDER'):
        print(f"🚀 Starting Job Tracker API on Render (port {port})")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print(f"🚀 Starting Job Tracker API locally (port {port})")
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
        app.run(debug=True, port=port, host='0.0.0.0')