import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function SignUpPage() {
  const [orgName, setOrgName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleSignUp = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      await axios.post('http://localhost:5555/v1/auth/signup', {
        org_name: orgName,
        email,
        password,
      });
      setMessage('✅ Registration successful! Redirecting to login...');
      setTimeout(() => navigate('/'), 1500);
    } catch (err) {
      setMessage('❌ Registration failed.');
    }
  };

  return (
    <div className="page">
      <h2>Sign Up</h2>
      <form onSubmit={handleSignUp}>
        <input
          type="text"
          placeholder="Organization Name"
          value={orgName}
          onChange={e => setOrgName(e.target.value)}
          required
        />
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />
        <button type="submit">Register</button>
      </form>
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default SignUpPage;
