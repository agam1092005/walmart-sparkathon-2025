import { Link } from 'react-router-dom';

function LandingPage() {
  return (
    <div className="page" style={{ textAlign: 'center' }}>
      <h2>Welcome to Quantum-Secure Fraud Detection</h2>
      <p style={{ margin: '20px 0', fontSize: '1.1rem' }}>
        Experience next-generation fraud detection powered by federated learning and quantum-safe encryption.
      </p>
      <div style={{ marginTop: '32px' }}>
        <Link to="/login">
          <button style={{ marginRight: '16px' }}>Login</button>
        </Link>
        <Link to="/signup">
          <button>Sign Up</button>
        </Link>
      </div>
    </div>
  );
}

export default LandingPage; 