import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, glow, rand, SectionTag, Typed, useSpring} from '../ui';
import {useCue} from '../cue';

const RAIN = 'the model may say this now or later spider France citrus leverage honestly BUT ethical'.split(' ');

export const Cold: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const titleAt = cue('Anthropic');
  const s = useSpring(titleAt, 18);
  const glitch = frame > titleAt && frame < titleAt + 12 ? (rand(frame) - 0.5) * 30 : 0;
  return (
    <AbsoluteFill>
      {Array.from({length: 38}).map((_, i) => {
        const x = rand(i) * 1920;
        const speed = 1 + rand(i + 7) * 2.5;
        const y = ((frame * speed + rand(i + 3) * 1200) % 1300) - 100;
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: x,
              top: y,
              fontSize: 22,
              color: C.faint,
              writingMode: 'vertical-rl',
            }}
          >
            {RAIN[i % RAIN.length]}
          </div>
        );
      })}
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column'}}>
        <div style={{fontSize: 40, color: C.dim, marginBottom: 40, height: 50}}>
          <Typed text="> where do thoughts wait before they are spoken?" start={10} cps={28} />
        </div>
        <div
          style={{
            fontSize: 220,
            letterSpacing: 30,
            color: C.green,
            textShadow: glow(C.green, 40),
            transform: `scale(${0.6 + 0.4 * s}) translateX(${glitch}px)`,
            opacity: s,
          }}
        >
          J-SPACE
        </div>
        <div style={{fontSize: 32, color: C.dim, letterSpacing: 6, opacity: s, marginTop: 20}}>
          THE VERBALIZABLE WORKSPACE INSIDE A LANGUAGE MODEL
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// Hundreds of silent computations; only a handful reach the output words.
export const Problem: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const lit = cue('special subset');
  const brain = cue('the way');
  const cols = 32;
  const rows = 14;
  const selected = new Set([37, 101, 150, 222, 260, 301, 355, 399]);
  return (
    <AbsoluteFill>
      <SectionTag n="01" label="THE QUESTION" />
      <div style={{position: 'absolute', top: 150, left: 0, right: 0, textAlign: 'center', fontSize: 44, color: C.white}}>
        {['output', 'words', 'the', 'model', 'writes'].map((w, i) => (
          <span key={w} style={{margin: '0 14px', opacity: interpolate(frame, [10 + i * 6, 25 + i * 6], [0, 1], {extrapolateRight: 'clamp'})}}>
            {w}
          </span>
        ))}
      </div>
      <svg width={1920} height={1080} style={{position: 'absolute'}}>
        {Array.from({length: cols * rows}).map((_, i) => {
          const cx = 260 + (i % cols) * 45;
          const cy = 300 + Math.floor(i / cols) * 40;
          const pulse = 0.25 + 0.35 * Math.abs(Math.sin(frame / 9 + i));
          const isSel = selected.has(i) && frame > lit;
          const up = isSel ? interpolate(frame, [lit, lit + 40], [0, 1], {extrapolateRight: 'clamp'}) : 0;
          return (
            <g key={i}>
              {isSel && (
                <line x1={cx} y1={cy} x2={960 + (cx - 960) * 0.3} y2={cy - (cy - 190) * up} stroke={C.amber} strokeWidth={2} opacity={0.8} />
              )}
              <circle cx={cx} cy={cy} r={isSel ? 9 : 5} fill={isSel ? C.amber : C.green} opacity={isSel ? 1 : pulse * 0.6} />
            </g>
          );
        })}
      </svg>
      <div
        style={{
          position: 'absolute',
          bottom: 170,
          left: 0,
          right: 0,
          textAlign: 'center',
          fontSize: 34,
          color: C.amber,
          opacity: interpolate(frame, [brain, brain + 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
        }}
      >
        GLOBAL WORKSPACE THEORY · "ACCESS CONSCIOUSNESS"
      </div>
    </AbsoluteFill>
  );
};
