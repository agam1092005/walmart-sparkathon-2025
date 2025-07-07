import products from "../data/products";
import ProductCard from "../components/ProductCard";

export default function Home() {
  return (
  <div
  style={{
    display: "flex",
    gap: "30px",
    padding: "60px 40px 40px", 
    justifyContent: "center",
    flexWrap: "wrap",
  }}
>
  {products.map((product) => (
    <ProductCard key={product.id} product={product} />
  ))}
</div>

  );
}
