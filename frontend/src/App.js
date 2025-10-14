// App.js - Simplified and fixed version
import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
// Detect API URL based on environment
const API_URL = window.location.hostname === 'localhost' 
  ? '${API_URL}' 
  : window.location.origin;
function App() {
  const [message, setMessage] = useState('');
  const [authStatus, setAuthStatus] = useState({ 
    authenticated: false, 
    email_processing: false,
    has_results: false,
    models_loaded: { classifier: false, ner: false }
  });
  const [results, setResults] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedEmail, setExpandedEmail] = useState(null);
  const [processingStarted, setProcessingStarted] = useState(false);

  // Check authentication status
  const checkAuthStatus = useCallback(async () => {
    try {
      const response = await fetch('${API_URL}/auth/status', {
        credentials: 'include'
      });
      const data = await response.json();
      setAuthStatus(data);
      
      // Show model status
      if (data.models_loaded) {
        console.log('Model Status:', data.models_loaded);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
    }
  }, []);

  // Fetch results
  const fetchResults = useCallback(async () => {
    try {
      console.log('Fetching results...');
      const response = await fetch('${API_URL}/api/results', {
        credentials: 'include'
      });
      
      if (!response.ok) {
        if (response.status === 404) {
          setMessage('No results found. Please process emails first.');
          return;
        }
        throw new Error(`Server error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Results loaded:', data);
      
      setResults(data.job_applications || []);
      setStats(data.stats || {});
      setMessage(`Loaded ${data.job_applications?.length || 0} job applications`);
    } catch (error) {
      console.error('Error fetching results:', error);
      setMessage('Error loading results: ' + error.message);
    }
  }, []);

  // Process emails
  const processEmails = async () => {
    if (processingStarted) return;
    
    try {
      setProcessingStarted(true);
      setLoading(true);
      setMessage('📧 Scanning your emails for job applications...');
      
      const response = await fetch('${API_URL}/api/process-emails', {
        credentials: 'include'
      });
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setMessage(`✅ Found ${data.applications} job applications in ${data.processed} emails!`);
        // Fetch the results automatically
        setTimeout(() => {
          fetchResults();
        }, 1000);
      } else {
        setMessage('Error: ' + (data.error || 'Processing failed'));
      }
    } catch (error) {
      setMessage('Error: ' + error.message);
    } finally {
      setLoading(false);
      setProcessingStarted(false);
    }
  };

  // Handle Google auth
  const handleGoogleAuth = async () => {
    try {
      setLoading(true);
      const response = await fetch('${API_URL}/auth/url', {
        credentials: 'include'
      });
      const data = await response.json();
      
      if (data.auth_url) {
        window.location.href = data.auth_url;
      } else {
        setMessage('Error: ' + (data.error || 'Failed to get auth URL'));
      }
    } catch (error) {
      setMessage('Error connecting to server: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Clear session
  const clearSession = async () => {
    try {
      const response = await fetch('${API_URL}/api/clear', {
        credentials: 'include'
      });
      const data = await response.json();
      setMessage(data.message);
      setResults(null);
      setStats(null);
      checkAuthStatus();
    } catch (error) {
      setMessage('Error clearing session: ' + error.message);
    }
  };

  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    try {
      // Try to parse and format the date
      const date = new Date(dateString);
      if (!isNaN(date.getTime())) {
        return date.toLocaleDateString('en-US', { 
          month: 'short', 
          day: 'numeric', 
          year: 'numeric' 
        });
      }
    } catch (e) {
      // If parsing fails, try to extract readable part
    }
    // Return first part of the string if it looks like a date
    const parts = dateString.split(' ');
    if (parts.length >= 3) {
      return parts.slice(0, 3).join(' ');
    }
    return dateString;
  };

  // Toggle email expansion
  const toggleEmailExpanded = (gmailId) => {
    setExpandedEmail(expandedEmail === gmailId ? null : gmailId);
  };

  // Get confidence level
  const getConfidenceLevel = (confidence) => {
    if (confidence > 0.8) return 'high';
    if (confidence > 0.5) return 'medium';
    return 'low';
  };

  // Initialize on mount and handle OAuth callback
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    
    if (urlParams.get('auth') === 'success') {
      setMessage('✅ Successfully authenticated with Google!');
      // Clean URL
      window.history.replaceState({}, document.title, window.location.pathname);
      
      // Start processing emails automatically after auth
      setTimeout(() => {
        processEmails();
      }, 2000);
    } else if (urlParams.get('auth') === 'error') {
      setMessage('Authentication failed: ' + (urlParams.get('message') || 'Unknown error'));
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    checkAuthStatus();
  }, []);

  // Auto-load results if available
  useEffect(() => {
    if (authStatus.authenticated && authStatus.has_results && !results && !loading) {
      fetchResults();
    }
  }, [authStatus, results, loading, fetchResults]);

  // Landing page for non-authenticated users
  if (!authStatus.authenticated) {
    return (
      <div className="App">
        <div className="hero-section">
          <div className="container">
            <nav className="navbar">
              <div className="nav-brand">
                <span className="logo">💼</span>
                <span className="brand-name">JobTracker AI</span>
              </div>
            </nav>
            
            <div className="hero-content">
              <h1 className="hero-title">
                Track Your <span className="gradient-text">Job Applications</span> Automatically
              </h1>
              <p className="hero-subtitle">
                Connect your Gmail and let AI scan for job applications, extract company names, 
                positions, and organize everything in one beautiful dashboard.
              </p>
              
              <div className="features-grid">
                <div className="feature-card">
                  <div className="feature-icon">🔍</div>
                  <h3>Smart Detection</h3>
                  <p>AI identifies job application emails automatically</p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon">🏢</div>
                  <h3>Company Extraction</h3>
                  <p>Extracts company names and positions from emails</p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon">📊</div>
                  <h3>Track Progress</h3>
                  <p>View all applications in one organized dashboard</p>
                </div>
              </div>
              
              <button 
                onClick={handleGoogleAuth} 
                className="cta-button"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Connecting...
                  </>
                ) : (
                  <>
                    <span>📧</span>
                    Connect Gmail & Get Started
                  </>
                )}
              </button>
              
              {message && (
                <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
                  {message}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Dashboard for authenticated users
  return (
    <div className="App">
      <div className="dashboard">
        <nav className="navbar">
          <div className="nav-brand">
            <span className="logo">💼</span>
            <span className="brand-name">JobTracker AI</span>
          </div>
          <div className="nav-actions">
            {!authStatus.models_loaded?.classifier && (
              <span className="warning-badge">⚠️ Classifier Offline</span>
            )}
            {!authStatus.models_loaded?.ner && (
              <span className="warning-badge">⚠️ NER Offline</span>
            )}
            <button 
              onClick={processEmails} 
              className="btn btn-primary"
              disabled={loading || authStatus.email_processing}
            >
              {loading ? (
                <>
                  <div className="spinner small"></div>
                  Processing...
                </>
              ) : (
                '📧 Scan Emails'
              )}
            </button>
            <button 
              onClick={fetchResults} 
              className="btn"
              disabled={loading}
            >
              🔄 Refresh
            </button>
            <button 
              onClick={clearSession} 
              className="btn btn-danger"
            >
              🗑️ Clear
            </button>
            <div className="auth-badge">
              ✅ Connected
            </div>
          </div>
        </nav>

        <div className="container">
          {message && (
            <div className={`banner-message ${message.includes('Error') || message.includes('No results') ? 'error' : 'success'}`}>
              {message}
            </div>
          )}

          {authStatus.email_processing && (
            <div className="processing-banner">
              <div className="spinner"></div>
              <span>Processing emails... Please wait</span>
            </div>
          )}

          {stats && results && (
            <div className="dashboard-content">
              <div className="stats-overview">
                <div className="stat-card">
                  <div className="stat-number">{results.length}</div>
                  <div className="stat-label">Total Applications</div>
                </div>
                <div className="stat-card">
                  <div className="stat-number">
                    {results.filter(app => app.company).length}
                  </div>
                  <div className="stat-label">Companies Found</div>
                </div>
                <div className="stat-card">
                  <div className="stat-number">
                    {results.filter(app => app.position).length}
                  </div>
                  <div className="stat-label">Positions Found</div>
                </div>
                <div className="stat-card">
                  <div className="stat-number">
                    {Math.round((results.filter(app => app.confidence > 0.8).length / results.length) * 100)}%
                  </div>
                  <div className="stat-label">High Confidence</div>
                </div>
              </div>

              <div className="applications-section">
                <h2>Your Job Applications</h2>
                <div className="applications-grid">
                  {results.map((app, index) => (
                    <div key={app.gmail_id || index} className="application-card">
                      <div className="card-header">
                        <div className="company-info">
                          <div className="company-avatar">
                            {app.company ? app.company.charAt(0).toUpperCase() : '?'}
                          </div>
                          <div className="company-details">
                            <h3>{app.company || 'Unknown Company'}</h3>
                            <p>{app.position || 'Position not specified'}</p>
                          </div>
                        </div>
                        <div className="application-meta">
                          <span className="date">{formatDate(app.date)}</span>
                          <span className={`confidence ${getConfidenceLevel(app.confidence)}`}>
                            {Math.round(app.confidence * 100)}% match
                          </span>
                        </div>
                      </div>
                      <div className="card-actions">
                        <button 
                          className="btn-text"
                          onClick={() => toggleEmailExpanded(app.gmail_id)}
                        >
                          {expandedEmail === app.gmail_id ? 'Hide Email' : 'View Email'}
                        </button>
                      </div>
                      {expandedEmail === app.gmail_id && (
                        <div className="email-preview">
                          <div className="email-header">
                            <div><strong>Subject:</strong> {app.subject}</div>
                            <div><strong>From:</strong> {app.from}</div>
                            <div><strong>Date:</strong> {formatDate(app.date)}</div>
                          </div>
                          <div className="email-body">
                            {app.body ? app.body.split('\n').map((para, idx) => 
                              para.trim() ? <p key={idx}>{para}</p> : <br key={idx} />
                            ) : 'No email content available'}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {!results && !loading && authStatus.has_results && (
            <div className="empty-state">
              <div className="empty-icon">📊</div>
              <h3>Results Available</h3>
              <p>Click the button below to load your previous scan results.</p>
              <button onClick={fetchResults} className="btn btn-primary">
                Load Results
              </button>
            </div>
          )}

          {!results && !loading && !authStatus.has_results && (
            <div className="empty-state">
              <div className="empty-icon">🚀</div>
              <h3>Ready to Start Tracking?</h3>
              <p>Connect your Gmail to scan for job applications and track your progress.</p>
              <button onClick={processEmails} className="btn btn-primary">
                Scan My Emails
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;