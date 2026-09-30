import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT, glow, rand} from '../ui';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// Decaying impulse: 1 at the hit frame, falling to 0 over `len` frames.
export const impulse = (frame: number, hits: number[], len = 12) =>
  hits.reduce((a, h) => (frame >= h && frame < h + len ? Math.max(a, 1 - (frame - h) / len) : a), 0);

// Camera shake from a list of hit frames.
export const Shake: React.FC<{hits: number[]; amp?: number; children: React.ReactNode}> = ({hits, amp = 26, children}) => {
  const frame = useCurrentFrame();
  const k = impulse(frame, hits, 14);
  const x = (rand(frame * 1.7) - 0.5) * 2 * amp * k;
  const y = (rand(frame * 3.1 + 9) - 0.5) * 2 * amp * k;
  const r = (rand(frame * 5.3) - 0.5) * 2 * k;
  return <AbsoluteFill style={{transform: `translate(${x}px, ${y}px) rotate(${r}deg) scale(${1 + 0.04 * k})`}}>{children}</AbsoluteFill>;
};

export const Flash: React.FC<{hits: number[]; color?: string; strength?: number}> = ({hits, color = C.green, strength = 0.55}) => {
  const frame = useCurrentFrame();
  const k = impulse(frame, hits, 8);
  return <AbsoluteFill style={{backgroundColor: color, opacity: k * strength, mixBlendMode: 'screen', pointerEvents: 'none'}} />;
};

// Backrooms corridor: perspective rings rushing toward the camera.
export const Corridor: React.FC<{speed?: number; color?: string; vpY?: number; boost?: number[]}> = ({
  speed = 1,
  color = C.green,
  vpY = 820,
  boost = [],
}) => {
  const frame = useCurrentFrame();
  const W = 1080;
  const H = 1920;
  const vpX = 540;
  const b = impulse(frame, boost, 20);
  const t = frame * 0.012 * speed * (1 + 3 * b);
  const N = 14;
  const rings = Array.from({length: N}).map((_, i) => {
    const z = (i / N + t) % 1; // 0 far -> 1 near
    const s = 0.02 + Math.pow(z, 2.6) * 1.9;
    return {s, o: Math.min(1, z * 1.6) * (1 - Math.max(0, z - 0.85) / 0.15)};
  });
  const rays = 16;
  return (
    <svg width={W} height={H} style={{position: 'absolute', inset: 0}}>
      {Array.from({length: rays}).map((_, i) => {
        const a = (i / rays) * Math.PI * 2 + 0.2;
        return (
          <line
            key={i}
            x1={vpX}
            y1={vpY}
            x2={vpX + Math.cos(a) * 2200}
            y2={vpY + Math.sin(a) * 2200}
            stroke={color}
            strokeOpacity={0.12}
            strokeWidth={2}
          />
        );
      })}
      {rings.map((r, i) => (
        <rect
          key={i}
          x={vpX - (W * 0.9 * r.s) / 2}
          y={vpY - (H * 0.62 * r.s) / 2}
          width={W * 0.9 * r.s}
          height={H * 0.62 * r.s}
          fill="none"
          stroke={color}
          strokeOpacity={0.45 * r.o}
          strokeWidth={1 + 3 * r.s}
        />
      ))}
      <circle cx={vpX} cy={vpY} r={90} fill={color} opacity={0.06 + 0.1 * b} />
    </svg>
  );
};

// Chromatic-aberration text that splits on hits and jitters slightly always.
export const GlitchText: React.FC<{
  text: string;
  size: number;
  color?: string;
  hits?: number[];
  style?: React.CSSProperties;
}> = ({text, size, color = C.green, hits = [], style}) => {
  const frame = useCurrentFrame();
  const k = impulse(frame, hits, 10);
  const idle = rand(Math.floor(frame / 3)) > 0.93 ? 0.35 : 0;
  const off = (k + idle) * size * 0.12;
  const skew = (k > 0 ? (rand(frame) - 0.5) * 20 : 0);
  const base: React.CSSProperties = {position: 'absolute', inset: 0, whiteSpace: 'nowrap'};
  return (
    <div style={{position: 'relative', fontSize: size, lineHeight: 1, fontFamily: FONT, letterSpacing: size * 0.04, ...style}}>
      <span style={{visibility: 'hidden', whiteSpace: 'nowrap'}}>{text}</span>
      <span style={{...base, color: '#ff2a6d', transform: `translate(${-off}px, 0)`, opacity: 0.8, mixBlendMode: 'screen'}}>{text}</span>
      <span style={{...base, color: '#05d9e8', transform: `translate(${off}px, 0)`, opacity: 0.8, mixBlendMode: 'screen'}}>{text}</span>
      <span style={{...base, color, textShadow: glow(color, size * 0.25), transform: `skewX(${skew}deg)`}}>{text}</span>
    </div>
  );
};

// Word that slams in from huge scale at frame `at`.
export const Slam: React.FC<{at: number; children: React.ReactNode; from?: number; style?: React.CSSProperties}> = ({
  at,
  children,
  from = 3.2,
  style,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (frame < at) return null;
  const s = spring({frame: frame - at, fps, config: {damping: 14, stiffness: 220, mass: 0.6}});
  const scale = interpolate(s, [0, 1], [from, 1]);
  const blur = interpolate(frame - at, [0, 5], [14, 0], clamp);
  const o = interpolate(frame - at, [0, 3], [0, 1], clamp);
  return <div style={{transform: `scale(${scale})`, filter: `blur(${blur}px)`, opacity: o, ...style}}>{children}</div>;
};

// Horizontal slice displacement for a few frames at scene start (glitch cut).
export const GlitchCut: React.FC<{len?: number}> = ({len = 7}) => {
  const frame = useCurrentFrame();
  if (frame >= len) return null;
  const k = 1 - frame / len;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      {Array.from({length: 9}).map((_, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: (rand(i + frame * 7) - 0.5) * 300 * k,
            top: rand(i * 13 + frame) * 1920,
            width: 1080,
            height: 20 + rand(i + 3) * 90,
            background: i % 3 === 0 ? C.green : i % 3 === 1 ? '#ff2a6d' : '#05d9e8',
            opacity: 0.35 * k,
          }}
        />
      ))}
      <AbsoluteFill style={{backgroundColor: '#fff', opacity: 0.25 * k}} />
    </AbsoluteFill>
  );
};

// Word particles bursting from a point.
export const Burst: React.FC<{at: number; words: string[]; x: number; y: number; color?: string; spread?: number}> = ({
  at,
  words,
  x,
  y,
  color = C.amber,
  spread = 420,
}) => {
  const frame = useCurrentFrame();
  const t = frame - at;
  if (t < 0 || t > 50) return null;
  return (
    <>
      {words.map((w, i) => {
        const a = rand(i + at) * Math.PI * 2;
        const d = interpolate(t, [0, 30], [0, spread * (0.6 + 0.4 * rand(i * 7 + at))], {...clamp, easing: (v) => 1 - Math.pow(1 - v, 3)});
        const o = interpolate(t, [0, 4, 35, 50], [0, 1, 1, 0], clamp);
        return (
          <div
            key={w + i}
            style={{
              position: 'absolute',
              left: x + Math.cos(a) * d,
              top: y + Math.sin(a) * d,
              transform: 'translate(-50%, -50%)',
              fontFamily: FONT,
              fontSize: 34 + 14 * rand(i + 2),
              color,
              textShadow: glow(color, 10),
              opacity: o,
            }}
          >
            {w}
          </div>
        );
      })}
    </>
  );
};

// TikTok-style kinetic captions: 3-word groups, active word pops.
export const KineticCaptions: React.FC<{text: string; seconds: number; lead: number; top?: number}> = ({
  text,
  seconds,
  lead,
  top = 1330,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const words = text.split(/\s+/);
  const total = text.length;
  let pos = 0;
  const timed = words.map((w) => {
    const i = text.indexOf(w, pos);
    pos = i + w.length;
    return {w, t: lead + (i / total) * seconds * fps};
  });
  let now = -1;
  timed.forEach((x, i) => {
    if (frame >= x.t) now = i;
  });
  if (now < 0 || frame > lead + seconds * fps + 10) return null;
  const g = Math.floor(now / 3) * 3;
  const group = timed.slice(g, g + 3);
  return (
    <div
      style={{
        position: 'absolute',
        top,
        left: 60,
        right: 140,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        gap: '0 22px',
        fontFamily: FONT,
        fontSize: 76,
        lineHeight: 1.15,
      }}
    >
      {group.map((x, i) => {
        const idx = g + i;
        const active = idx === now;
        const seen = idx <= now;
        const pop = active ? interpolate(frame - x.t, [0, 4], [1.25, 1], clamp) : 1;
        return (
          <span
            key={idx}
            style={{
              color: active ? C.bg : seen ? C.white : 'rgba(216,255,228,0.35)',
              background: active ? C.green : 'transparent',
              padding: '0 10px',
              transform: `scale(${pop})`,
              display: 'inline-block',
              textShadow: active ? 'none' : '0 0 10px #000, 0 0 4px #000',
              boxShadow: active ? `0 0 24px ${C.green}` : undefined,
            }}
          >
            {x.w.toUpperCase()}
          </span>
        );
      })}
    </div>
  );
};
