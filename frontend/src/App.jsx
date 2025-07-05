import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { useEffect, useRef, useState, createContext } from 'react';
import LoginPage from './pages/Login';
import SignUpPage from './pages/SignUp';
import UploadPage from './pages/UploadPage';
import Dashboard from './pages/Dashboard';
import ProtectedRoute from './components/ProtectedRoute';
import RootRedirect from './components/RootRedirect';
import AuthRedirect from './components/AuthRedirect';
import 'locomotive-scroll/dist/locomotive-scroll.css';

export const LocoScrollContext = createContext({ scrollRef: null, locomotiveInstance: null });

function ScrollProvider({ children }) {
  const scrollRef = useRef(null);
  const [locomotiveInstance, setLocomotiveInstance] = useState(null);
  const location = useLocation();

  useEffect(() => {
    let scrollInstance;
    import('locomotive-scroll').then((LocomotiveScroll) => {
      scrollInstance = new LocomotiveScroll.default({
        el: scrollRef.current,
        smooth: true,
        multiplier: 0.6,
      });
      setLocomotiveInstance(scrollInstance);
    });
    return () => {
      if (scrollInstance) scrollInstance.destroy();
    };
  }, []);

  // Update/scroll to top on route change
  useEffect(() => {
    if (locomotiveInstance) {
      locomotiveInstance.scrollTo(0, { duration: 0 });
      setTimeout(() => locomotiveInstance.update(), 100);
    }
  }, [location, locomotiveInstance]);

  return (
    <LocoScrollContext.Provider value={{ scrollRef, locomotiveInstance }}>
      <div ref={scrollRef} data-scroll-container style={{ minHeight: '100vh' }}>
        {children}
      </div>
    </LocoScrollContext.Provider>
  );
}

function App() {
  return (
    <Router>
      {/* <Header /> */}
      <ScrollProvider>
        <Routes>
          <Route path="/" element={<RootRedirect />} />
          <Route path="/login" element={<AuthRedirect><LoginPage /></AuthRedirect>} />
          <Route path="/signup" element={<AuthRedirect><SignUpPage /></AuthRedirect>} />
          <Route path="/upload" element={<ProtectedRoute><UploadPage /></ProtectedRoute>} />
          <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        </Routes>
      </ScrollProvider>
    </Router>
  );
}

export default App;