import { Link } from "react-router-dom";

function Header() {
  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        left: 0,
        width: "100%",
        height:"40px",
        background: "#ffffff",
        padding: "16px 18px",
        boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
        display: "flex",
        justifyContent: "center", 
        alignItems: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          fontWeight: "bold",
          fontSize: "25px",
          color: "#333",
          position: "absolute",
          left: "50%",
          transform: "translateX(-50%)",
        }}
      >
        🛍️ TEST SHOPPING
      </div>

      <Link
        to="/"
        style={{
          position: "absolute",
          left: "32px",
          padding: "8px 16px",
          backgroundColor: "#ffffff",
          color: "#007bff",
          borderRadius: "5px",
          textDecoration: "none",
          fontSize: "16px",
          fontWeight: "500",
        }}
      >
        🏠 Home
      </Link>
    </nav>
  );
}

export default Header;
