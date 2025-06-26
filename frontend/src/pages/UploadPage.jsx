import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function UploadPage() {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  const handleUpload = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('dataset', file);
    try {
      await axios.post('http://localhost:5000/upload', formData);
      navigate('/result');
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  return (
    <div className="page">
      <h2>Upload Dataset</h2>
      <form onSubmit={handleUpload}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          required
        />
        <button type="submit">Submit Dataset</button>
      </form>
    </div>
  );
}

export default UploadPage;
