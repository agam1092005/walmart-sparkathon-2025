import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function SignUpPage() {
  const [newUser, setNewUser] = useState('');
  const [newPass, setNewPass] = useState('');
  const navigate = useNavigate();

  const handleSignUp = (e) => {
    e.preventDefault();
    if (newUser && newPass) {
      localStorage.setItem('user', newUser);
      navigate('/');
    }
  };

  return (
    <div className="page">
      <h2>Sign Up</h2>
      <form onSubmit={handleSignUp}>
        <input
          type="text"
          placeholder="New Username"
          value={newUser}
          onChange={(e) => setNewUser(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="New Password"
          value={newPass}
          onChange={(e) => setNewPass(e.target.value)}
          required
        />
        <button type="submit">Register</button>
      </form>
    </div>
  );
}

export default SignUpPage;
