import { useState } from "react";

function Payment() {
  const [selectedMethod, setSelectedMethod] = useState("");
  const [isPaid, setIsPaid] = useState(false);

  const handlePayment = () => {
    if (!selectedMethod) {
      alert("Please select a payment method.");
      return;
    }
    setIsPaid(true);
  };

  const paymentMethods = [
    "UPI (Google Pay, PhonePe, Paytm)",
    "Credit Card",
    "Debit Card",
    "Net Banking",
    "Wallet (Paytm, Amazon Pay)",
  ];

  return (
    <div style={{ padding: "40px", maxWidth: "500px", margin: "0 auto", textAlign: "center" }}>
      <h2>💳 Choose Payment Method</h2>

      {paymentMethods.map((method) => (
        <label
          key={method}
          style={{
            display: "block",
            margin: "12px 0",
            textAlign: "left",
            padding: "10px",
            border: selectedMethod === method ? "2px solid #007bff" : "1px solid #ccc",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          <input
            type="radio"
            name="payment"
            value={method}
            checked={selectedMethod === method}
            onChange={(e) => setSelectedMethod(e.target.value)}
            style={{ marginRight: "10px" }}
          />
          {method}
        </label>
      ))}

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

      {isPaid && (
        <div style={{ marginTop: "30px", color: "green", fontWeight: "bold" }}>
          ✅ Payment Successful via {selectedMethod}!
        </div>
      )}
    </div>
  );
}

export default Payment;
