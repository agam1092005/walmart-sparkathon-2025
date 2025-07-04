import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      const res = await axios.post(
        'http://localhost:5555/v1/auth/signin',
        { email, password },
        { withCredentials: true }
      );
      if (res.data.token) {
        setMessage(`✅ Welcome, ${res.data.user.org_name}`);
        setTimeout(() => navigate('/dashboard'), 1500);
      } else {
        setMessage('❌ Invalid credentials');
      }
    } catch (err) {
      setMessage('❌ Server error. Try again later.');
    }
  };

  return (
    <div className="page">
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
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
        <button type="submit">Login</button>
      </form>
      <p>New user? <Link to="/signup">Sign Up</Link></p>
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default LoginPage;
