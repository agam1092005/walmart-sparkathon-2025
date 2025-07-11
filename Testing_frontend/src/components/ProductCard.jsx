import { Link } from "react-router-dom";

function ProductCard({ product, onViewDetails }) {
  return (
    <div
      style={{
        border: "1px solid gray",
        padding: "20px",
        width: "250px",
        textAlign: "center",
        boxSizing: "border-box",
        borderRadius: "8px",
        boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      <img
        src={product.image}
        alt={product.name}
        style={{
          width: "150px",
          height: "150px",
          objectFit: "contain",
          marginBottom: "10px",
        }}
      />
      <h3>{product.name}</h3>
      <p>₹{product.price}</p>
      <Link to={`/product/${product.id}`} style={{ color: "blue" }} onClick={onViewDetails}>
        View Details
      </Link>
    </div>
  );
}

export default ProductCard;
