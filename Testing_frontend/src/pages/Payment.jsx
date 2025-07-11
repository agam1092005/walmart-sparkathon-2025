import { useState, useRef } from "react";
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

function Payment() {
  const [cardNumber, setCardNumber] = useState("");
  const [expiry, setExpiry] = useState("");
  const [cvv, setCvv] = useState("");
  const [isPaid, setIsPaid] = useState(false);
  const [botDetected, setBotDetected] = useState(false);
  const [warning, setWarning] = useState("");
  const [error, setError] = useState("");

  // For velocity & behavioral checks
  const lastPaymentTime = useRef([]); // array of timestamps
  const inputChangeTimes = useRef([]); // array of timestamps
  const invalidAttemptTimes = useRef([]); // array of timestamps for invalid attempts

  // Helper to get current time
  const now = () => Date.now();

  // Behavioral: Track input changes
  const handleInputChange = (setter) => (e) => {
    setter(e.target.value);
    inputChangeTimes.current.push(now());
    if (inputChangeTimes.current.length > 10) inputChangeTimes.current.shift();
  };

  // Rate limiting & velocity checks
  const handlePayment = () => {
    setError("");
    const currentTime = now();
    // Validate card details
    if (
      cardNumber.trim() !== "0000 0000 0000 0000" ||
      expiry.trim() !== "01/2030" ||
      cvv.trim() !== "000"
    ) {
      // Track invalid attempts
      invalidAttemptTimes.current.push(currentTime);
      if (invalidAttemptTimes.current.length > 10) invalidAttemptTimes.current.shift();
      // Check for fraud: >3 invalid attempts in 1 minute
      const oneMinuteAgo = currentTime - 60 * 1000;
      const invalidsLastMinute = invalidAttemptTimes.current.filter(ts => ts > oneMinuteAgo).length;
      if (invalidsLastMinute > 3) {
        setBotDetected(true);
        setWarning("Suspicious activity detected: Too many invalid card attempts in a short time.");
        setIsPaid(false);
        incrementAmazonDetected();
        return;
      }
      setError("Invalid card details. Please enter the correct test card.");
      return;
    }
    lastPaymentTime.current.push(currentTime);
    if (lastPaymentTime.current.length > 10) lastPaymentTime.current.shift();

    // --- Bot detection logic ---
    let suspicious = false;
    let reasons = [];

    // 1. Too many payment attempts in 1 minute
    const oneMinuteAgo = currentTime - 60 * 1000;
    const attemptsLastMinute = lastPaymentTime.current.filter(ts => ts > oneMinuteAgo).length;
    if (attemptsLastMinute > 2) {
      suspicious = true;
      reasons.push("Too many payment attempts in a short time");
    }

    // 2. Superhuman speed: time between last two attempts < 1s
    if (lastPaymentTime.current.length >= 2) {
      const t1 = lastPaymentTime.current[lastPaymentTime.current.length - 1];
      const t2 = lastPaymentTime.current[lastPaymentTime.current.length - 2];
      if (t1 - t2 < 1000) {
        suspicious = true;
        reasons.push("Payment attempts too fast (superhuman speed)");
      }
    }

    // 3. Rapid input changes: >5 changes in last 30 seconds
    const thirtySecondsAgo = currentTime - 30 * 1000;
    const changesLast30s = inputChangeTimes.current.filter(ts => ts > thirtySecondsAgo).length;
    if (changesLast30s > 5) {
      suspicious = true;
      reasons.push("Too many card detail changes in a short time");
    }

    setIsPaid(true);
    setBotDetected(suspicious);
    setWarning(suspicious ? `Bot activity detected: ${reasons.join(", ")}` : "");
    if (suspicious) {
      incrementAmazonDetected();
    }
  };

  return (
    <div style={{ padding: "40px", maxWidth: "500px", margin: "0 auto", textAlign: "center" }}>
      <h2>💳 Enter Card Details</h2>
      <div style={{ margin: "20px 0", textAlign: "left" }}>
        <label style={{ display: "block", marginBottom: "10px" }}>
          Card Number
          <input
            type="text"
            value={cardNumber}
            onChange={handleInputChange(setCardNumber)}
            placeholder="0000 0000 0000 0000"
            style={{ width: "100%", padding: "8px", marginTop: "4px", borderRadius: "4px", border: "1px solid #ccc" }}
            maxLength={19}
          />
        </label>
        <label style={{ display: "block", marginBottom: "10px" }}>
          Expiry (MM/YYYY)
          <input
            type="text"
            value={expiry}
            onChange={handleInputChange(setExpiry)}
            placeholder="01/2030"
            style={{ width: "100%", padding: "8px", marginTop: "4px", borderRadius: "4px", border: "1px solid #ccc" }}
            maxLength={7}
          />
        </label>
        <label style={{ display: "block", marginBottom: "10px" }}>
          CVV
          <input
            type="password"
            value={cvv}
            onChange={handleInputChange(setCvv)}
            placeholder="000"
            style={{ width: "100%", padding: "8px", marginTop: "4px", borderRadius: "4px", border: "1px solid #ccc" }}
            maxLength={3}
          />
        </label>
      </div>
      <button
        onClick={handlePayment}
        style={{
          marginTop: "20px",
          padding: "12px 24px",
          fontSize: "16px",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Pay Now
      </button>
      {error && (
        <div style={{ color: "red", marginTop: "16px", fontWeight: "bold" }}>{error}</div>
      )}
      {botDetected && warning && (
        <div style={{ marginTop: "30px", color: "red", fontWeight: "bold" }}>
          🚨 {warning}
        </div>
      )}
      {isPaid && !botDetected && (
        <div style={{ marginTop: "30px", color: "green", fontWeight: "bold" }}>
          ✅ Payment Successful via Credit Card!
        </div>
      )}
    </div>
  );
}

export default Payment;
