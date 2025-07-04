import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { hasValidToken } from '../utils/auth';

function AuthRedirect({ children }) {
  const navigate = useNavigate();

  useEffect(() => {
    if (hasValidToken()) {
      navigate('/dashboard');
    }
  }, [navigate]);

  // If has valid token, don't render anything (will redirect)
  if (hasValidToken()) {
    return null;
  }

  // If no valid token, render the children (login/signup component)
  return children;
}

export default AuthRedirect; 