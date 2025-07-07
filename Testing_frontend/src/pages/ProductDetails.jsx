import { useParams, useNavigate } from "react-router-dom";
import products from "../data/products";

function ProductDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const product = products.find((p) => p.id === parseInt(id));

  return (
    <div style={{ padding: "40px", textAlign: "center" }}>
      <img
        src={product.image}
        alt={product.name}
        style={{
          width: "250px",
          height: "250px",
          objectFit: "contain",
          marginBottom: "20px",
        }}
      />
      <h2>{product.name}</h2>
      <p>₹{product.price}</p>
      <p>{product.description}</p>
      
      <button
        onClick={() => navigate("/payment")}
        style={{
          padding: "10px 20px",
          marginTop: "20px",
          cursor: "pointer",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px"
        }}
      >
        Buy Now
      </button>
    </div>
  );
}

export default ProductDetail;
