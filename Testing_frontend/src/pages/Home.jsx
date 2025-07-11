import products from "../data/products";
import ProductCard from "../components/ProductCard";
import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import PocketBase from 'pocketbase';
const pb = new PocketBase('http://127.0.0.1:8090'); // Change if needed

async function incrementAmazonDetected() {
  try {
    const records = await pb.collection('data').getFullList({
      filter: 'company="Test"'
    });
    if (records.length > 0) {
      const record = records[0];
      await pb.collection('data').update(record.id, {
        detected: (record.detected || 0) + 1
      });
    }
  } catch (err) {
    console.error('Failed to update PocketBase:', err);
  }
}

export default function Home() {
  const [botDetected, setBotDetected] = useState(false);
  const [warning, setWarning] = useState("");
  const navigate = useNavigate();
  const landingTime = useRef(null);
  const openCount = useRef(0);

  useEffect(() => {
    landingTime.current = window.performance.now();
    openCount.current = 0;
    setBotDetected(false);
    setWarning("");
  }, []);

  const handleProductOpen = (id) => (e) => {
    const now = window.performance.now();
    if (landingTime.current && now - landingTime.current < 2000) {
      openCount.current += 1;
      if (openCount.current > 2) {
        setBotDetected(true);
        setWarning("Bot activity detected: Too many product tabs opened instantly");
        incrementAmazonDetected();
        e.preventDefault();
        return;
      }
    }
    // else allow navigation
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "30px",
        padding: "60px 40px 40px",
        justifyContent: "center",
        flexWrap: "wrap",
        flexDirection: "column"
      }}
    >
      {botDetected && (
        <div style={{ color: "red", fontWeight: "bold", marginBottom: "20px", textAlign: "center" }}>
          🚨 {warning}
        </div>
      )}
      <div style={{ display: "flex", gap: "30px", flexWrap: "wrap", justifyContent: "center" }}>
        {products.map((product) => (
          <div key={product.id}>
            <ProductCard product={product} onViewDetails={handleProductOpen(product.id)} />
          </div>
        ))}
      </div>
    </div>
  );
}
