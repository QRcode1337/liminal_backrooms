import React, {createContext, useContext} from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

// Phosphor CRT palette, matching the liminal_backrooms GUI.
export const C = {
  bg: '#030805',
  green: '#33ff66',
  dim: '#1a7a38',
  faint: '#0d3a1c',
  amber: '#ffb000',
  red: '#ff4d4d',
  white: '#d8ffe4',
};

export const FONT = 'Iosevka Term';

export const FontFace: React.FC = () => (
  <style>{`@font-face{font-family:'${FONT}';src:url('${staticFile(
    'iosevkaterm-regular.ttf',
  )}') format('truetype');}`}</style>
);

export const glow = (color: string, px = 12) =>
  `0 0 ${px / 3}px ${color}, 0 0 ${px}px ${color}`;

export const useSpring = (delay = 0, damping = 200) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: {damping}});
};

export const fadeIn = (frame: number, start: number, len = 15) =>
  interpolate(frame, [start, start + len], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

// Seconds -> frames helper for scene timelines.
export const useSec = () => {
  const {fps} = useVideoConfig();
  return (s: number) => Math.round(s * fps);
};

// Deterministic pseudo-random.
export const rand = (seed: number) => {
  const x = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
};

export const Screen: React.FC<{children: React.ReactNode}> = ({children}) => (
  <AbsoluteFill
    style={{
      backgroundColor: C.bg,
      fontFamily: FONT,
      color: C.green,
    }}
  >
    {children}
  </AbsoluteFill>
);

export const CRT: React.FC = () => {
  const frame = useCurrentFrame();
  const flicker = 0.035 + 0.02 * rand(frame);
  const rollY = (frame * 4) % 1300;
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <AbsoluteFill
        style={{
          background:
            'repeating-linear-gradient(0deg, rgba(0,0,0,0.28) 0px, rgba(0,0,0,0.28) 1px, transparent 2px, transparent 4px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: rollY - 200,
          height: 160,
          background:
            'linear-gradient(180deg, transparent, rgba(51,255,102,0.04), transparent)',
        }}
      />
      <AbsoluteFill style={{backgroundColor: `rgba(51,255,102,${flicker})`}} />
      <AbsoluteFill
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.75) 100%)',
        }}
      />
    </AbsoluteFill>
  );
};

// True inside the vertical (TikTok) composition, where chrome like section tags
// is drawn outside the scaled scene instead.
export const VerticalCtx = createContext(false);

export const SectionTag: React.FC<{n: string; label: string}> = ({n, label}) => {
  const frame = useCurrentFrame();
  const vertical = useContext(VerticalCtx);
  const chars = Math.floor(interpolate(frame, [0, 20], [0, label.length], {extrapolateRight: 'clamp'}));
  if (vertical) return null;
  return (
    <div
      style={{
        position: 'absolute',
        top: 56,
        left: 80,
        fontSize: 26,
        letterSpacing: 4,
        color: C.dim,
      }}
    >
      <span style={{color: C.green}}>[{n}]</span> {label.slice(0, chars)}
      <span style={{opacity: frame % 30 < 15 ? 1 : 0}}>_</span>
    </div>
  );
};

// Split narration into caption units: whole sentences, or short word chunks
// (maxWords) that never cross a sentence boundary.
const captionParts = (text: string, maxWords?: number) => {
  const sentences = text.match(/[^.?!:]+[.?!:]+/g) ?? [text];
  if (!maxWords) return sentences;
  return sentences.flatMap((s) => {
    const words = s.trim().split(/\s+/);
    const n = Math.ceil(words.length / maxWords);
    const size = Math.ceil(words.length / n);
    return Array.from({length: n}, (_, i) => words.slice(i * size, (i + 1) * size).join(' ') + ' ');
  });
};

// Subtitles, timed by each unit's character share of the narration.
export const Captions: React.FC<{
  text: string;
  seconds: number;
  lead: number;
  maxWords?: number;
  style?: React.CSSProperties;
}> = ({text, seconds, lead, maxWords, style}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const parts = captionParts(text, maxWords);
  const total = parts.reduce((a, p) => a + p.length, 0);
  const t = (frame - lead) / fps;
  let acc = 0;
  let current = '';
  for (const p of parts) {
    const start = (acc / total) * seconds;
    acc += p.length;
    const end = (acc / total) * seconds;
    if (t >= start && t < end + (maxWords ? 0 : 0.3)) current = p.trim();
  }
  if (t < 0 || t > seconds + 0.3) current = '';
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 52,
        left: 160,
        right: 160,
        textAlign: 'center',
        fontSize: 34,
        lineHeight: 1.35,
        color: C.white,
        textShadow: '0 0 8px #000, 0 0 4px #000',
        ...style,
      }}
    >
      {current && (
        <span style={{backgroundColor: 'rgba(0,0,0,0.6)', padding: '6px 14px', boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone'}}>
          {current}
        </span>
      )}
    </div>
  );
};

export const Typed: React.FC<{text: string; start: number; cps?: number; style?: React.CSSProperties}> = ({
  text,
  start,
  cps = 40,
  style,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const n = Math.max(0, Math.floor(((frame - start) / fps) * cps));
  const shown = text.slice(0, n);
  return (
    <span style={style}>
      {shown}
      {n < text.length && n > 0 ? <span style={{opacity: 0.8}}>▌</span> : null}
    </span>
  );
};

// A glowing token chip, used for J-lens readouts.
export const Chip: React.FC<{word: string; color?: string; scale?: number; style?: React.CSSProperties}> = ({
  word,
  color = C.amber,
  scale = 1,
  style,
}) => (
  <span
    style={{
      display: 'inline-block',
      border: `2px solid ${color}`,
      color,
      padding: '6px 16px',
      fontSize: 34 * scale,
      textShadow: glow(color, 10),
      boxShadow: `0 0 14px ${color}55, inset 0 0 10px ${color}33`,
      background: 'rgba(0,0,0,0.5)',
      ...style,
    }}
  >
    {word}
  </span>
);
