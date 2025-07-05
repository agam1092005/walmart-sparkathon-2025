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
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [companyName, setCompanyName] = useState('');
  const [breachData, setBreachData] = useState([]);
  const [breachLoading, setBreachLoading] = useState(false);
  const [clientCount, setClientCount] = useState(0);
  const [detectedThreats, setDetectedThreats] = useState(0);
  const intervalRef = useRef();
  const navigate = useNavigate();

  const fetchClientCount = async () => {
    try {
      const res = await fetch('http://localhost:5555/v1/ml/clients_count', {
        credentials: 'include'
      });
      
      if (res.ok) {
        const data = await res.json();
        setClientCount(data.num_clients || 0);
      } else {
        console.error('Failed to fetch client count');
        setClientCount(0);
      }
    } catch (error) {
      console.error('Error fetching client count:', error);
      setClientCount(0);
    }
  };

  const fetchBreachData = async (company) => {
    if (!company || company === 'Unknown Company') return;
    
    setBreachLoading(true);
    try {
      const res = await fetch('http://localhost:5555/v1/data_breach/search_local', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ company_name: company })
      });
      
      if (res.ok) {
        const data = await res.json();
        setBreachData(data.results || []);
      } else {
        console.error('Failed to fetch breach data');
        setBreachData([]);
      }
    } catch (error) {
      console.error('Error fetching breach data:', error);
      setBreachData([]);
    } finally {
      setBreachLoading(false);
    }
  };

  useEffect(() => {
    let stopped = false;
    async function fetchStatus() {
      try {
        console.log('[DEBUG] Fetching status from /v1/ml/status');
        const res = await fetch('http://localhost:5555/v1/ml/status', {
          credentials: 'include'
        });
        console.log('[DEBUG] Response status:', res.status);
        if (!res.ok) {
          if (res.status === 401) {
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
        const company = data.org_name ? data.org_name : 'Unknown Company';
        setCompanyName(company);
        setDetectedThreats(data.detected || 0);
        setLoading(false);
        
        if (company !== 'Unknown Company') {
          fetchBreachData(company);
        }
        
        if (data.encrypted && intervalRef.current) {
          console.log('[DEBUG] Encrypted is true, stopping polling');
          clearInterval(intervalRef.current);
          intervalRef.current = null;
          fetchClientCount();
        }
        
        if (data.submitted && !data.encrypted && !intervalRef.current) {
          console.log('[DEBUG] Dataset submitted, starting polling');
          intervalRef.current = setInterval(fetchStatus, 5000);
        }
      } catch (e) {
        console.error('Error fetching status:', e);
        setLoading(false);
        setStatus(null);
        setCompanyName('Unknown Company');
      }
    }
    
    fetchStatus();
    fetchClientCount();
    
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
          {loading ? 'Loading...' : (companyName ? `${companyName} Dashboard` : 'Dashboard')}
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
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'left',
        justifyContent: 'flex-start',
        minHeight: 'calc(100vh - 80px)',
        padding: '0 20px'
      }}>
        <div style={{ 
          marginTop: '70px',
          textAlign: 'left',
          width: '100%',
          maxWidth: '600px'
        }}>
          <p style={{ margin: '20px 0px', fontSize: '1.1rem' }}>
              Welcome back! View your security stats from here.
          </p>
         
        </div>

        {loading ? (
          <div style={{ marginTop: '50px', fontSize: '1.1rem', color: '#666' }}>
            Loading status...
          </div>
        ) : (
          <>
            <div style={{ 
              display: 'flex', 
              gap: '20px', 
              marginBottom: 24
            }}>
              <div style={{ 
                padding: '40px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.08)',
                flex: '1',
                textAlign: 'left',
                minWidth: '300px'
              }}>
                <StatusDot color={dotColor} />
                <span style={{ fontWeight: 500, fontSize: '1.1rem' }}>{statusMsg}</span>
                {subMsg && (
                  <div style={{ fontSize: '0.95rem', color: '#888', marginTop: 8 }}>{subMsg}</div>
                )}
                {showUpload && (
                  <div style={{ marginTop: '20px' }}>
                    <Link to="/upload">
                      <button style={{ 
                        background: '#00bcd4', 
                        color: '#fff', 
                        border: 'none', 
                        borderRadius: 6, 
                        padding: '10px 22px', 
                        fontWeight: 600, 
                        fontSize: '1rem', 
                        cursor: 'pointer',
                        transition: 'background 0.2s ease'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = '#00a0b0';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = '#00bcd4';
                      }}
                      >
                        Upload Dataset
                      </button>
                    </Link>
                  </div>
                )}
              </div>

              <div style={{ 
                padding: '30px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.08)',
                flex: '1',
                textAlign: 'left',
                minWidth: '200px'
              }}>
                <span style={{ fontWeight: 500, fontSize: '1rem' }}>Clients Contributing</span>
                <div style={{ fontSize: '2rem', color: '#4caf50', marginTop: 8, fontWeight: 'bold' }}>
                  {clientCount}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#888', marginTop: 4 }}>
                  Active participants
                </div>
              </div>

              <div style={{ 
                padding: '30px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.08)',
                flex: '1',
                textAlign: 'left',
                minWidth: '200px'
              }}>
                <span style={{ fontWeight: 500, fontSize: '1rem' }}>Bots & Suspicion Detected</span>
                <div style={{ fontSize: '2rem', color: '#ff6b6b', marginTop: 8, fontWeight: 'bold' }}>
                  {detectedThreats}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#888', marginTop: 4 }}>
                  Threats identified
                </div>
              </div>
            </div>
           

            {companyName && companyName !== 'Unknown Company' && (
              <div style={{ 
                marginBottom: 24,
                padding: '40px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.08)',
                width: '100%',
                textAlign: 'left'
              }}>
                <h3 style={{ 
                  color: '#00bcd4', 
                  marginBottom: '20px', 
                  fontSize: '1.3rem',
                  fontWeight: 600 
                }}>
                  Data Breach Information for {companyName}
                </h3>
                
                {breachLoading ? (
                  <div style={{ color: '#888', fontSize: '1rem' }}>
                    Loading breach data...
                  </div>
                ) : breachData.length > 0 ? (
                  <div>
                    <p style={{ 
                      color: '#ff6b6b', 
                      fontSize: '1rem', 
                      marginBottom: '20px',
                      fontWeight: 500 
                    }}>
                      ⚠️ Found {breachData.length} breach record(s) for this company
                    </p>
                    
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr 1fr 1fr',
                      gap: '10px',
                      padding: '15px',
                      background: 'rgba(255, 107, 107, 0.2)',
                      borderRadius: '8px 8px 0 0',
                      border: '1px solid rgba(255, 107, 107, 0.3)',
                      borderBottom: 'none',
                      fontWeight: 'bold',
                      fontSize: '0.9rem'
                    }}>
                      <div style={{ color: '#ff6b6b' }}>Breach ID</div>
                      <div style={{ color: '#00bcd4' }}>Date</div>
                      <div style={{ color: '#00bcd4' }}>Domain</div>
                      <div style={{ color: '#00bcd4' }}>Records</div>
                      <div style={{ color: '#00bcd4' }}>Data Type</div>
                      <div style={{ color: '#00bcd4' }}>Industry</div>
                      <div style={{ color: '#00bcd4' }}>Risk</div>
                    </div>
                    
                    {breachData.map((breach, index) => (
                      <div key={index} style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr 1fr 1fr',
                        gap: '10px',
                        padding: '15px',
                        background: index % 2 === 0 ? 'rgba(255, 107, 107, 0.05)' : 'rgba(255, 107, 107, 0.1)',
                        border: '1px solid rgba(255, 107, 107, 0.3)',
                        borderTop: 'none',
                        fontSize: '0.9rem',
                        lineHeight: '1.3'
                      }}>
                        <div style={{ 
                          color: '#ff6b6b', 
                          fontWeight: '500',
                          wordBreak: 'break-word'
                        }}>
                          {breach.breach_id}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.breach_date}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.domain}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.exposed_records}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.exposed_data}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.industry}
                        </div>
                        <div style={{ wordBreak: 'break-word' }}>
                          {breach.password_risk}
                        </div>
                      </div>
                    ))}
                    
                    {breachData.map((breach, index) => (
                      <div key={`desc-${index}`} style={{
                        background: 'rgba(0, 0, 0, 0.2)',
                        border: '1px solid rgba(255, 107, 107, 0.3)',
                        borderRadius: '0 0 8px 8px',
                        padding: '15px',
                        marginBottom: '15px',
                        fontSize: '0.9rem',
                        lineHeight: '1.4'
                      }}>
                        <div style={{ 
                          color: '#ffd93d', 
                          fontWeight: 'bold',
                          marginBottom: '8px'
                        }}>
                          Description for {breach.breach_id}:
                        </div>
                        <div style={{ color: '#e0e0e0' }}>
                          {breach.description}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ 
                    color: '#4caf50', 
                    fontSize: '1rem',
                    padding: '15px',
                    background: 'rgba(76, 175, 80, 0.1)',
                    border: '1px solid rgba(76, 175, 80, 0.3)',
                    borderRadius: '8px'
                  }}>
                    ✅ No breach records found for {companyName}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard; 