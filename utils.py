# utils.py
import os
import re
import time
import pandas as pd
import base64
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import html
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline, TextClassificationPipeline
)
import torch

class EmailFetcher:
    def __init__(self, credentials):
        self.service = build('gmail', 'v1', credentials=credentials)
    
    def clean_email_body(self, text):
        """Clean and normalize email body text"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Fix common mojibake issues
        text = text.replace('‚Äô', "'")
        text = text.replace('‚Äò', "'")
        text = text.replace('‚Äú', '"')
        text = text.replace('‚Äù', '"')
        text = text.replace('‚Äì', '-')
        text = text.replace('‚Äî', '--')
        text = text.replace('â€¦', '...')
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()
        
        return text

    def get_email_body(self, payload):
        """Recursively extract and clean text from all parts of the email."""
        if 'parts' in payload:
            parts_text = []
            for part in payload['parts']:
                part_text = self.get_email_body(part)
                if part_text:
                    parts_text.append(part_text)
            return "\n".join(parts_text)
        
        mime_type = payload.get('mimeType', '')
        body_data = payload.get('body', {}).get('data')
        
        if body_data:
            try:
                decoded = base64.urlsafe_b64decode(body_data).decode('utf-8', errors='ignore')
                if mime_type == 'text/html':
                    soup = BeautifulSoup(decoded, 'html.parser')
                    text = soup.get_text(separator="\n")
                    return self.clean_email_body(text)
                elif mime_type == 'text/plain':
                    return self.clean_email_body(decoded)
            except Exception as e:
                print(f"Error decoding body: {e}")
                return ""
        return ""

    def fetch_email(self, msg_id, max_retries=4):
        """Fetch email metadata + body with improved text cleaning."""
        attempt = 0
        wait = 1
        while attempt < max_retries:
            try:
                m = self.service.users().messages().get(
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
                    attachment_id = body.get('attachmentId')

                    if data:
                        try:
                            decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                            mime_type = part.get('mimeType', '')
                            
                            if mime_type == 'text/html':
                                soup = BeautifulSoup(decoded, 'html.parser')
                                decoded = soup.get_text(separator="\n")
                            elif mime_type == 'text/plain':
                                pass
                            
                            texts.append(self.clean_email_body(decoded))
                        except Exception as e:
                            print(f"Error decoding part: {e}")

                    elif attachment_id:
                        mime_type = part.get('mimeType', '')
                        if mime_type and mime_type.startswith('text/'):
                            try:
                                att = self.service.users().messages().attachments().get(
                                    userId='me',
                                    messageId=msg_id,
                                    id=attachment_id
                                ).execute()
                                decoded = base64.urlsafe_b64decode(att['data']).decode('utf-8', errors='ignore')
                                if mime_type == 'text/html':
                                    soup = BeautifulSoup(decoded, 'html.parser')
                                    decoded = soup.get_text(separator="\n")
                                texts.append(self.clean_email_body(decoded))
                            except Exception as e:
                                print(f"Error fetching attachment: {e}")

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
                print(f"Error fetching email for {msg_id} (attempt {attempt}/{max_retries}): {e}")
                time.sleep(wait)
                wait *= 2

        return {
            'gmail_id': msg_id,
            'thread_id': None,
            'subject': '',
            'from': '',
            'date': '',
            'snippet': '',
            'internal_date': '',
            'body': ''
        }

    def fetch_emails(self):
        """EXACT replication of your working email fetching code"""
        try:
            # Build date range (one year ago -> today) - EXACT same as your working code
            today = datetime.today().date()
            one_year_ago = (today - timedelta(days=365)).strftime("%Y/%m/%d")
            tomorrow = (today + timedelta(days=1)).strftime("%Y/%m/%d")

            query = f"after:{one_year_ago} before:{tomorrow} subject:(application OR applying)"

            # Fetch message IDs (paged)
            all_messages = []
            page_token = None

            print("Listing message IDs with query:", query)
            while True:
                try:
                    results = self.service.users().messages().list(
                        userId='me',
                        q=query,
                        maxResults=500,
                        pageToken=page_token
                    ).execute()
                except HttpError as e:
                    print(f"Error fetching message IDs: {e}. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                msgs = results.get('messages', [])
                all_messages.extend(msgs)
                page_token = results.get('nextPageToken')
                if not page_token:
                    break

            print(f"Total message IDs fetched: {len(all_messages)}")

            # Fetch metadata + body for each message ID
            emails = []

            for i, msg in enumerate(all_messages):
                print(f"Processing email {i+1}/{len(all_messages)}")
                row = self.fetch_email(msg['id'])
                emails.append(row)

            df = pd.DataFrame(emails)
            print(f"✅ Successfully fetched {len(df)} emails")
            return df

        except Exception as e:
            print(f"Error in fetch_emails: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

class JobInformationExtractor:
    def __init__(self, model_path="./job_ner_model/checkpoint-6090"):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Loading NER model from {model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=self.device
            )
            print("✅ NER Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading NER model: {e}")
            raise
    
    def extract_information(self, email_body, subject=""):
        """Extract company and position from email text - EXACT same as your working code"""
        # Combine subject and body for context
        full_text = f"Subject: {subject}\n\n{email_body}"
        
        # Limit text length to avoid token limits
        full_text = full_text[:2000]  # First 2000 characters
        
        try:
            # Run NER
            entities = self.ner_pipeline(full_text)
            
            # Group entities by type
            companies = []
            positions = []
            
            for entity in entities:
                if entity['entity_group'] == 'COMPANY':
                    companies.append(entity['word'])
                elif entity['entity_group'] == 'POSITION':
                    positions.append(entity['word'])
            
            # Clean and deduplicate
            companies = self._clean_entities(companies)
            positions = self._clean_entities(positions)
            
            return {
                'companies': companies,
                'positions': positions,
                'primary_company': companies[0] if companies else None,
                'primary_position': positions[0] if positions else None,
                'entity_count': len(companies) + len(positions)
            }
        except Exception as e:
            print(f"Error in extraction: {e}")
            return {
                'companies': [],
                'positions': [],
                'primary_company': None,
                'primary_position': None,
                'entity_count': 0,
                'error': str(e)
            }
    
    def _clean_entities(self, entities):
        """Clean and deduplicate entities - EXACT same as your working code"""
        cleaned = []
        seen = set()
        
        for entity in entities:
            # Remove extra whitespace and normalize
            clean_entity = ' '.join(entity.split()).strip()
            
            # Remove common prefixes/suffixes from tokenization
            clean_entity = re.sub(r'^##', '', clean_entity)
            
            # Filter out very short entities (likely false positives)
            if len(clean_entity) > 2 and clean_entity not in seen:
                cleaned.append(clean_entity)
                seen.add(clean_entity)
        
        return cleaned
    
    def process_emails_batch(self, emails_df):
        """Process a batch of emails - EXACT same logic as your working code"""
        results = []
        
        print(f"Processing {len(emails_df)} emails with NER...")
        
        for idx, row in emails_df.iterrows():
            if idx % 10 == 0:  # Progress indicator
                print(f"Processed {idx}/{len(emails_df)} emails...")
            
            try:
                extraction = self.extract_information(
                    email_body=str(row['body']),
                    subject=str(row['subject'])
                )
                
                results.append({
                    'gmail_id': row['gmail_id'],
                    'subject': row['subject'],
                    'from': row.get('from', ''),
                    'company': extraction['primary_company'],
                    'position': extraction['primary_position'],
                    'all_companies': extraction['companies'],
                    'all_positions': extraction['positions'],
                    'entity_count': extraction['entity_count'],
                    'confidence': 'high' if extraction['entity_count'] > 0 else 'low'
                })
                
            except Exception as e:
                print(f"Error processing email {row['gmail_id']}: {e}")
                results.append({
                    'gmail_id': row['gmail_id'],
                    'subject': row['subject'],
                    'from': row.get('from', ''),
                    'company': None,
                    'position': None,
                    'all_companies': [],
                    'all_positions': [],
                    'entity_count': 0,
                    'confidence': 'error',
                    'error': str(e)
                })
        
        return pd.DataFrame(results)

class EmailClassifier:
    def __init__(self, model_path="./email_classifier_final2"):
        print(f"Loading classifier from {model_path}...")
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Classifier loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            raise
    
    def classify_emails(self, emails_df):
        """EXACT same classification logic as your test_classifier_on_emails.py"""
        print(f"📧 Classifying {len(emails_df)} emails...")
        
        results = []
        
        for idx, row in emails_df.iterrows():
            email_text = f"Subject: {row['subject']}\n\n{row['body'][:1000]}"  # Limit body length
            
            try:
                classification = self.classifier(email_text)[0]
                
                results.append({
                    'gmail_id': row['gmail_id'],
                    'subject': row['subject'],
                    'from': row.get('from', ''),
                    'date': row.get('date', ''),
                    'predicted_label': classification['label'],
                    'confidence': classification['score'],
                    'is_application': classification['label'] == 'JOB_APPLICATION'
                })
                
                if idx % 10 == 0:
                    print(f"Classified {idx}/{len(emails_df)}...")
                    
            except Exception as e:
                print(f"Error classifying email {row['gmail_id']}: {e}")
                results.append({
                    'gmail_id': row['gmail_id'],
                    'subject': row['subject'],
                    'from': row.get('from', ''),
                    'date': row.get('date', ''),
                    'predicted_label': 'ERROR',
                    'confidence': 0.0,
                    'is_application': False,
                    'error': str(e)
                })
        
        results_df = pd.DataFrame(results)
        application_count = len(results_df[results_df['is_application'] == True])
        
        print(f"\n📊 Classification Results:")
        print(f"✅ JOB_APPLICATION emails: {application_count}/{len(emails_df)}")
        print(f"❌ NON_APPLICATION emails: {len(emails_df) - application_count}/{len(emails_df)}")
        print(f"📈 Application rate: {(application_count/len(emails_df))*100:.1f}%")
        
        return results_df

class CompleteJobTracker:
    def __init__(self):
        print("🚀 Initializing Complete Job Tracker...")
        self.classifier = EmailClassifier()
        self.ner_extractor = JobInformationExtractor()
        print("✅ All models loaded successfully!")
    
    def process_emails(self, emails_df):
        """Complete pipeline: classify emails, then extract info from job applications"""
        print("Starting complete email processing pipeline...")
        
        # Stage 1: Classify emails
        classified_df = self.classifier.classify_emails(emails_df)
        
        # Get only job applications
        job_applications = classified_df[classified_df['is_application'] == True]
        print(f"🎯 Found {len(job_applications)} job applications")
        
        if len(job_applications) == 0:
            print("No job applications found!")
            return pd.DataFrame()
        
        # Merge with original emails to get full data for NER
        job_emails_with_data = pd.merge(
            job_applications, 
            emails_df[['gmail_id', 'body', 'snippet']], 
            on='gmail_id', 
            how='left'
        )
        
        # Stage 2: Extract job information from applications
        print("Extracting job information from applications...")
        ner_results = self.ner_extractor.process_emails_batch(job_emails_with_data)
        
        # Merge classification and NER results
        final_results = pd.merge(
            classified_df,
            ner_results[['gmail_id', 'company', 'position', 'all_companies', 'all_positions', 'entity_count']],
            on='gmail_id',
            how='left'
        )
        
        # Add email_type based on classification
        final_results['email_type'] = final_results['predicted_label']
        
        print(f"✅ Processing complete! Processed {len(final_results)} total emails")
        
        # Show summary
        apps_with_companies = len(final_results[(final_results['is_application'] == True) & (final_results['company'].notna())])
        apps_with_positions = len(final_results[(final_results['is_application'] == True) & (final_results['position'].notna())])
        
        print(f"📊 Job applications with companies found: {apps_with_companies}")
        print(f"📊 Job applications with positions found: {apps_with_positions}")
        
        return final_results