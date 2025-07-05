import { useContext, useState } from 'react';
import { Link } from 'react-router-dom';
import ScrollReveal from '../components/ScrollReveal';
import TrueFocus from '../components/TrueFocus';
import { LocoScrollContext } from '../App';
import PixelTransition from '../components/PixelTransition';
import companiesGif from '../assets/companies.gif';

function LandingPage() {
  const { scrollRef, locomotiveInstance } = useContext(LocoScrollContext);
  const [openFaqIndex, setOpenFaqIndex] = useState(null);

  const handleFaqToggle = (index) => {
    setOpenFaqIndex(openFaqIndex === index ? null : index);
  };

  return (
    <div>
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'transparent',
      }}>
        <div style={{ width: '100%', textAlign: 'center', marginBottom: '32px' }}>
          <span style={{ fontWeight: 900, color: '#fff', fontSize: '4.2rem', letterSpacing: '0.5px', lineHeight: 1.3 }}>
            Secure Federated Fraud Detection<br />
            <span style={{ fontWeight: 900 }}>for Online Shopping</span>
          </span>
        </div>
      
        <PixelTransition
          firstContent={
            <img
              src={companiesGif}
              alt="Logos of companies participating in federated learning."
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
          }
          secondContent={
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: "#000"
              }}
            >
              <p style={{ fontWeight: 900, fontSize: "2rem", color: "#ffffff", padding: "10px", textAlign: "center", margin: 0 }}>
                Train together, share nothing.
              </p>
            </div>
          }
          gridSize={12}
          pixelColor='#000'
          animationStepDuration={0.4}
          className="custom-pixel-card"
        />
        <div style={{ display: 'flex', flexDirection: 'row', gap: '24px', marginTop: '32px' }}>
          <Link to="/login">
            <button style={{ color: 'white' }}>Login</button>
          </Link>
          <Link to="/signup">
            <button style={{ color: 'white' }}>Sign Up</button>
          </Link>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: '100%', padding: '0 80px', boxSizing: 'border-box' }}>
          <ScrollReveal
            baseOpacity={0}
            enableBlur={true}
            baseRotation={5}
            blurStrength={10}
            scrollContainerRef={scrollRef}
            locomotiveInstance={locomotiveInstance}
          >
            Online shopping fraud is a rising storm—affecting every retailer, every day. Yet, most are still fighting alone, relying on isolated models and outdated defenses. What if we could unite retailers to build a smarter fraud detection system. System built on collaboration.  But what about privacy of our customers?
          </ScrollReveal>
        </div>
      </div>
      {/* FAQ Section */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', margin: '32px 0' }}>
        <div style={{ width: '100%', maxWidth: '600px' }}>
          {[{
            q: 'What is federated learning?',
            a: 'Federated learning is a collaborative machine learning approach where data remains decentralized and only model updates are shared.'
          }, {
            q: 'How is my data kept private?',
            a: 'Your data never leaves your device or organization. Only encrypted model updates are shared.'
          }, {
            q: 'Who can participate in the training?',
            a: 'Any retailer or organization interested in improving fraud detection can join the federated network.'
          }, {
            q: 'Is this system secure?',
            a: 'Yes, all communications and model updates are encrypted to ensure maximum security.'
          }].map((faq, idx) => (
            <details 
              key={idx} 
              open={openFaqIndex === idx}
              onClick={(e) => {
                e.preventDefault();
                handleFaqToggle(idx);
              }}
              style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '12px', color: '#fff', fontSize: '0.95rem', padding: '12px 18px', border: '1px solid rgba(255,255,255,0.12)' }}
            >
              <summary style={{ cursor: 'pointer', fontWeight: 600, outline: 'none' }}>{faq.q}</summary>
              <div style={{ marginTop: '8px', color: '#fff', fontSize: '0.92rem', fontWeight: 400 }}>{faq.a}</div>
            </details>
          ))}
        </div>
      </div>
      <TrueFocus 
          sentence="Customer data never leaves your walls."
          manualMode={false}
          blurAmount={6}
          borderColor="cyan"
          animationDuration={0.5}
          pauseBetweenAnimations={1}
        />
    </div>
  );
}

export default LandingPage; 