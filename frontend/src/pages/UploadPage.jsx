import { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { LocoScrollContext } from '../App';

function UploadPage() {
  const { scrollRef, locomotiveInstance } = useContext(LocoScrollContext);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleUpload = async (e) => {
    e.preventDefault();
    setMessage('');
    const formData = new FormData();
    formData.append('dataset', file);
    try {
      await axios.post(
        'http://localhost:5555/v1/ml/upload',
        formData,
        { withCredentials: true }
      );
      setMessage('✅ Upload successful! Training started.');
      setTimeout(() => navigate('/dashboard'), 1500);
    } catch (error) {
      setMessage('❌ Upload failed.');
    }
  };

  return (
    <div className="page">
      <h2>Upload Dataset</h2>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept=".csv"
          onChange={e => setFile(e.target.files[0])}
          required
        />
        <button type="submit">Submit Dataset</button>
      </form>
      {message && <p className="message">{message}</p>}
    </div>
  );
}

export default UploadPage;
