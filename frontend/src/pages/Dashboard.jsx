import { useContext, useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { LocoScrollContext } from '../App';
import { clearAuthCookies } from '../utils/auth';

function StatusDot({ color }) {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 12,
        height: 12,
        borderRadius: '50%',
        background: color,
        marginRight: 8,
        verticalAlign: 'middle',
        boxShadow: '0 0 4px ' + color,
      }}
    />
  );
}

function Dashboard() {
  const { scrollRef, locomotiveInstance } = useContext(LocoScrollContext);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [companyName, setCompanyName] = useState('');
  const intervalRef = useRef();
  const navigate = useNavigate();

  useEffect(() => {
    let stopped = false;
    async function fetchStatus() {
      try {
        console.log('[DEBUG] Fetching status from /v1/ml/status');
        const res = await fetch('/v1/ml/status', {
          credentials: 'include' // Include cookies
        });
        console.log('[DEBUG] Response status:', res.status);
        if (!res.ok) {
          if (res.status === 401) {
            // Token expired or invalid, redirect to login
            console.log('[DEBUG] 401 error, redirecting to login');
            clearAuthCookies();
            navigate('/login');
            return;
          }
          throw new Error('Failed to fetch status');
        }
        const data = await res.json();
        console.log('[DEBUG] Status data received:', data);
        setStatus(data);
        setCompanyName(data.org_name ? data.org_name : 'Unknown Company');
        setLoading(false);
        // Stop polling if encrypted is true (global model is ready)
        if (data.encrypted && intervalRef.current) {
          console.log('[DEBUG] Encrypted is true, stopping polling');
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } catch (e) {
        console.error('Error fetching status:', e);
        setLoading(false);
        setStatus(null);
        setCompanyName('Unknown Company');
      }
    }
    
    fetchStatus();
    if (!intervalRef.current) {
      intervalRef.current = setInterval(fetchStatus, 5000); // Poll every 5 seconds
    }
    
    return () => {
      stopped = true;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [navigate]);

  let dotColor = 'gray';
  let statusMsg = 'No dataset uploaded.';
  let subMsg = '';
  let showUpload = true;
  
  if (status) {
    if (status.encrypted) {
      dotColor = 'green';
      statusMsg = 'Using global model';
      subMsg = 'Your data and model are encrypted for customer privacy.';
      showUpload = false;
    } else if (status.hasTrained) {
      dotColor = 'yellow';
      statusMsg = 'Local model trained, waiting to connect with global model...';
      subMsg = 'Encrypting for customer privacy.';
      showUpload = false;
    } else if (status.submitted) {
      dotColor = 'red';
      statusMsg = 'Dataset uploaded, training in progress...';
      subMsg = '';
      showUpload = false;
    } else {
      dotColor = 'gray';
      statusMsg = 'No dataset uploaded.';
      showUpload = true;
    }
  }

  function handleLogout() {
    clearAuthCookies();
    navigate('/login');
  }

  return (
    <div style={{ minHeight: '100vh', background: 'none' }}>
      {/* Navbar at the very top */}
      <div style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 32px 12px 32px',
        borderBottom: '2px solid #00bcd4',
        background: 'rgba(30, 30, 30, 0.9)',
        backdropFilter: 'blur(6px)',
        position: 'fixed',
        top: 0,
        left: 0,
        zIndex: 10,
        boxShadow: '0 2px 10px rgba(0, 188, 212, 0.1)',
      }}>
        <div style={{ fontWeight: 700, fontSize: '1.25rem', letterSpacing: 0.5, color: '#00bcd4' }}>
          {loading ? 'Loading...' : (companyName || 'Unknown Company')}
        </div>
        <button
          onClick={handleLogout}
          style={{
            background: 'linear-gradient(135deg, #f44336, #d32f2f)',
            color: '#fff',
            border: 'none',
            borderRadius: 10,
            padding: '8px 18px',
            fontWeight: 600,
            fontSize: '1rem',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(244, 67, 54, 0.3)',
            transition: 'transform 0.2s ease, background 0.3s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.transform = 'translateY(-2px)';
            e.target.style.background = 'linear-gradient(135deg, #d32f2f, #c62828)';
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.background = 'linear-gradient(135deg, #f44336, #d32f2f)';
          }}
        >
          Logout
        </button>
      </div>
      {/* Main content below navbar */}
      <div style={{

        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        minHeight: 'calc(100vh - 80px)',
        padding: '0 20px'
      }}>
        {/* Company Name Header */}
        <div style={{ 
             marginTop: '100px',
          marginBottom: '32px', 
          textAlign: 'center',
          padding: '20px',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '12px',
          border: '1px solid rgba(255,255,255,0.1)',
          width: '100%',
          maxWidth: '600px'
        }}>
          <h1 style={{ 
            margin: 0, 
            fontSize: '2rem', 
            fontWeight: 700,
            color: '#00bcd4',
            textTransform: 'uppercase',
            letterSpacing: '1px'
          }}>
            {companyName}
          </h1>
        </div>

        {loading ? (
          <div style={{ marginTop: '50px', fontSize: '1.1rem', color: '#666' }}>
            Loading status...
          </div>
        ) : (
          <>
            <div style={{ 
              marginBottom: 24,
              padding: '20px',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.08)',
              width: '100%',
              maxWidth: '600px',
              textAlign: 'center'
            }}>
              <StatusDot color={dotColor} />
              <span style={{ fontWeight: 500, fontSize: '1.1rem' }}>{statusMsg}</span>
              {subMsg && (
                <div style={{ fontSize: '0.95rem', color: '#888', marginTop: 8 }}>{subMsg}</div>
              )}
            </div>
            <p style={{ margin: '20px 0', fontSize: '1.1rem', textAlign: 'center' }}>
              Welcome to your dashboard! What would you like to do next?
            </p>
            {showUpload && (
              <div style={{ marginTop: '32px', textAlign: 'center' }}>
                <Link to="/upload">
                  <button style={{ background: '#00bcd4', color: '#fff', border: 'none', borderRadius: 6, padding: '10px 22px', fontWeight: 600, fontSize: '1rem', cursor: 'pointer' }}>Upload Dataset</button>
                </Link>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 