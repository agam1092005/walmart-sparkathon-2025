import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { hasValidToken } from '../utils/auth';
import LandingPage from '../pages/LandingPage';

function RootRedirect() {
  const navigate = useNavigate();

  useEffect(() => {
    if (hasValidToken()) {
      navigate('/dashboard');
    }
  }, [navigate]);

  // If no valid token, show landing page
  if (!hasValidToken()) {
    return <LandingPage />;
  }

  // If has valid token, don't render anything (will redirect)
  return null;
}

export default RootRedirect; 