import { useEffect, useRef, useMemo } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

import './ScrollReveal.css';

gsap.registerPlugin(ScrollTrigger);

const ScrollReveal = ({
  children,
  scrollContainerRef,
  locomotiveInstance,
  enableBlur = true,
  baseOpacity = 0.1,
  baseRotation = 5,
  blurStrength = 20,
  containerClassName = "",
  textClassName = "",
  rotationEnd = "bottom bottom",
  wordAnimationEnd = "bottom bottom"
}) => {
  const containerRef = useRef(null);

  const splitText = useMemo(() => {
    const text = typeof children === 'string' ? children : '';
    return text.split(/(\s+)/).map((word, index) => {
      if (word.match(/^\s+$/)) return word;
      return (
        <span className="word" key={index}>
          {word}
        </span>
      );
    });
  }, [children]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let scroller = window;
    if (scrollContainerRef && scrollContainerRef.current) {
      scroller = scrollContainerRef.current;
    }

    let updateST;
    let triggers = [];

    // If using Locomotive Scroll, set up scrollerProxy only when instance is ready
    if (locomotiveInstance && scroller !== window) {
      ScrollTrigger.scrollerProxy(scroller, {
        scrollTop(value) {
          return arguments.length
            ? locomotiveInstance.scrollTo(value, { duration: 0, disableLerp: true })
            : locomotiveInstance.scroll.instance.scroll.y;
        },
        getBoundingClientRect() {
          return { top: 0, left: 0, width: window.innerWidth, height: window.innerHeight };
        },
        pinType: scroller.style.transform ? 'transform' : 'fixed',
      });

      updateST = () => ScrollTrigger.update();
      locomotiveInstance.on('scroll', updateST);
    }

    // Setup GSAP animations
    triggers.push(
      gsap.fromTo(
        el,
        { transformOrigin: '0% 50%', rotate: baseRotation },
        {
          ease: 'none',
          rotate: 0,
          scrollTrigger: {
            trigger: el,
            scroller,
            start: 'top bottom',
            end: rotationEnd,
            scrub: true,
          },
        }
      )
    );

    const wordElements = el.querySelectorAll('.word');

    triggers.push(
      gsap.fromTo(
        wordElements,
        { opacity: baseOpacity, willChange: 'opacity' },
        {
          ease: 'none',
          opacity: 1,
          stagger: 0.05,
          scrollTrigger: {
            trigger: el,
            scroller,
            start: 'top bottom-=20%',
            end: wordAnimationEnd,
            scrub: true,
          },
        }
      )
    );

    if (enableBlur) {
      triggers.push(
        gsap.fromTo(
          wordElements,
          { filter: `blur(${blurStrength}px)` },
          {
            ease: 'none',
            filter: 'blur(0px)',
            stagger: 0.05,
            scrollTrigger: {
              trigger: el,
              scroller,
              start: 'top bottom-=20%',
              end: wordAnimationEnd,
              scrub: true,
            },
          }
        )
      );
    }

    // Always refresh after setup
    setTimeout(() => {
      ScrollTrigger.refresh();
    }, 100);

    return () => {
      if (locomotiveInstance && updateST) {
        locomotiveInstance.off('scroll', updateST);
      }
      triggers.forEach(anim => {
        if (anim.scrollTrigger) anim.scrollTrigger.kill();
        anim.kill();
      });
      ScrollTrigger.getAll().forEach(trigger => trigger.kill());
      if (locomotiveInstance && scroller !== window) {
        ScrollTrigger.scrollerProxy(scroller, null);
      }
    };
  // Only run when locomotiveInstance is ready (or not used), and other dependencies
  }, [scrollContainerRef, enableBlur, baseRotation, baseOpacity, rotationEnd, wordAnimationEnd, blurStrength, locomotiveInstance]);

  return (
    <h2 ref={containerRef} className={`scroll-reveal ${containerClassName}`}>
      <p className={`scroll-reveal-text ${textClassName}`}>{splitText}</p>
    </h2>
  );
};

export default ScrollReveal; 