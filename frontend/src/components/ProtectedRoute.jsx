import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { hasValidToken } from '../utils/auth';

function ProtectedRoute({ children }) {
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(null);

  useEffect(() => {
    const checkAuth = () => {
      const hasToken = hasValidToken();
      setIsAuthenticated(hasToken);
      if (!hasToken) {
        navigate('/login');
      }
    };
    checkAuth();
    const timeoutId = setTimeout(checkAuth, 100);
    return () => clearTimeout(timeoutId);
  }, [navigate]);

  if (isAuthenticated === null) {
    return <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      color: '#fff',
      fontSize: '1.1rem'
    }}>Loading...</div>;
  }

  if (!isAuthenticated) {
    return null;
  }

  return children;
}

export default ProtectedRoute; 