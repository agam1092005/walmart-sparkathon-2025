import { Link } from 'react-router-dom';

function Dashboard() {
  return (
    <div className="page" style={{ textAlign: 'center' }}>
      <h2>Dashboard</h2>
      <p style={{ margin: '20px 0', fontSize: '1.1rem' }}>
        Welcome to your dashboard! What would you like to do next?
      </p>
      <div style={{ marginTop: '32px' }}>
        <Link to="/upload">
          <button>Upload Dataset</button>
        </Link>
      </div>
    </div>
  );
}

export default Dashboard; 