import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';

function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage(''); // Clear previous message

    try {
      const res = await axios.post('http://localhost:5000/login', {
        username,
        password,
      });

      if (res.data.success) {
        localStorage.setItem('user', username);
        setMessage(`✅ Welcome, ${username}`);
        setTimeout(() => navigate('/upload'), 1500);
      } else {
        setMessage('❌ Invalid credentials');
      }
    } catch (err) {
      console.error(err);
      setMessage('❌ Server error. Try again later.');
    }
  };

  return (
    <div className="page">
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <input
          type="text"
          placeholder="Login ID"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
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
