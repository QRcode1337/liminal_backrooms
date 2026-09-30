import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, fadeIn, glow, rand, SectionTag, useSpring} from '../ui';
import {useCue} from '../cue';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// Grid of residual-stream cells: layers (rows) x token positions (cols).
export const Lens: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tFuture = cue('not just');
  const tAvg = cue('averaged');
  const tUnembed = cue('Multiply');
  const tVec = cue('a vector');
  const L = 10;
  const P = 9;
  const x0 = 160;
  const y0 = 200;
  const dx = 78;
  const dy = 52;
  const srcL = 5;
  const srcP = 2;
  const cellX = (p: number) => x0 + p * dx;
  const cellY = (l: number) => y0 + (L - 1 - l) * dy;
  const vocab = [
    ['spider', 0.92],
    ['web', 0.61],
    ['eight', 0.48],
    ['legs', 0.33],
    ['silk', 0.21],
  ] as const;
  return (
    <AbsoluteFill>
      <SectionTag n="02" label="THE JACOBIAN LENS" />
      <svg width={1920} height={1080} style={{position: 'absolute'}}>
        <text x={x0 - 30} y={y0 + L * dy + 10} fill={C.dim} fontSize={22} fontFamily="Iosevka Term">
          layer ↑ / token position →
        </text>
        {Array.from({length: L * P}).map((_, i) => {
          const l = Math.floor(i / P);
          const p = i % P;
          const isSrc = l === srcL && p === srcP;
          const isTarget = l === L - 1 && p >= srcP;
          const targetOn = isTarget && frame > (p === srcP ? 0 : tFuture);
          return (
            <rect
              key={i}
              x={cellX(p) - 28}
              y={cellY(l) - 18}
              width={56}
              height={36}
              fill={isSrc ? C.amber : targetOn ? C.green : 'transparent'}
              fillOpacity={isSrc ? 0.9 : targetOn ? 0.35 : 0}
              stroke={isSrc ? C.amber : C.faint}
              strokeWidth={2}
            />
          );
        })}
        {Array.from({length: P - srcP}).map((_, k) => {
          const p = srcP + k;
          const on = k === 0 ? fadeIn(frame, 10) : fadeIn(frame, tFuture + k * 5);
          return (
            <path
              key={p}
              d={`M ${cellX(srcP)} ${cellY(srcL) - 18} Q ${(cellX(srcP) + cellX(p)) / 2} ${cellY(L - 1) - 120} ${cellX(p)} ${cellY(L - 1) - 18}`}
              stroke={C.amber}
              strokeWidth={2}
              fill="none"
              opacity={on * 0.85}
              strokeDasharray="6 6"
              strokeDashoffset={-frame}
            />
          );
        })}
        <text x={cellX(srcP) - 30} y={cellY(srcL) + 50} fill={C.amber} fontSize={24} fontFamily="Iosevka Term">
          h_ℓ
        </text>
        <text x={cellX(P - 1) - 60} y={cellY(L - 1) - 40} fill={C.green} fontSize={24} fontFamily="Iosevka Term" opacity={fadeIn(frame, tFuture)}>
          all t' ≥ t
        </text>
      </svg>
      {/* averaging counter */}
      <div style={{position: 'absolute', left: x0, top: y0 + L * dy + 50, fontSize: 28, color: C.dim, opacity: fadeIn(frame, tAvg)}}>
        E over{' '}
        <span style={{color: C.green}}>{Math.floor(interpolate(frame, [tAvg, tAvg + 45], [1, 1000], clamp))}</span> prompts → J_ℓ
      </div>
      {/* formula + readout */}
      <div style={{position: 'absolute', left: 1020, top: 220, width: 800}}>
        <div style={{fontSize: 30, color: C.dim}}>J_ℓ = E[ ∂h_final,t' / ∂h_ℓ,t ]</div>
        <div
          style={{
            fontSize: 44,
            color: C.green,
            marginTop: 30,
            textShadow: glow(C.green, 14),
            opacity: fadeIn(frame, tUnembed),
          }}
        >
          lens(h) = softmax(W_U · norm(J_ℓ h))
        </div>
        <div style={{marginTop: 50, opacity: fadeIn(frame, tUnembed + 15)}}>
          {vocab.map(([w, v], i) => {
            const grow = interpolate(frame, [tUnembed + 20 + i * 6, tUnembed + 50 + i * 6], [0, v as number], clamp);
            return (
              <div key={w} style={{display: 'flex', alignItems: 'center', marginBottom: 14, fontSize: 30}}>
                <div style={{width: 130, color: i === 0 ? C.amber : C.green}}>{w}</div>
                <div style={{height: 26, width: 520 * grow, background: i === 0 ? C.amber : C.dim, boxShadow: i === 0 ? `0 0 14px ${C.amber}` : undefined}} />
              </div>
            );
          })}
        </div>
        <div style={{marginTop: 30, fontSize: 30, color: C.amber, opacity: fadeIn(frame, tVec)}}>
          row of W_U·J_ℓ = "the model may say this, now or later"
        </div>
      </div>
      {/* sparkle so the grid feels alive */}
      {Array.from({length: 8}).map((_, i) => (
        <div key={i} style={{position: 'absolute', left: 160 + rand(i + frame * 0.01) * 700, top: 200 + rand(i * 3) * 480, width: 3, height: 3, background: C.green, opacity: 0.3}} />
      ))}
    </AbsoluteFill>
  );
};

// J-space = sparse non-negative combination of ~25 J-lens vectors.
export const Define: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tSparse = cue('sparse');
  const tSmall = cue('It is small');
  const tVar = cue('It explains');
  const tPoised = cue('But it is');
  const N = 180;
  const active = new Set(Array.from({length: 25}).map((_, i) => Math.floor(rand(i + 40) * N)));
  const rot = frame * 0.12;
  const pct = interpolate(frame, [tVar, tVar + 40], [0, 9.6], clamp);
  const s = useSpring(tPoised, 20);
  return (
    <AbsoluteFill>
      <SectionTag n="03" label="DEFINING J-SPACE" />
      <svg width={1920} height={1080} style={{position: 'absolute'}}>
        <g transform={`translate(640 560) rotate(${rot})`}>
          {Array.from({length: N}).map((_, i) => {
            const a = (i / N) * Math.PI * 2;
            const isOn = active.has(i) && frame > tSparse + (i % 25) * 1.5;
            const len = 300 + rand(i) * 80;
            return (
              <line
                key={i}
                x1={0}
                y1={0}
                x2={Math.cos(a) * len}
                y2={Math.sin(a) * len}
                stroke={isOn ? C.amber : C.faint}
                strokeWidth={isOn ? 3 : 1}
                opacity={isOn ? 1 : 0.7}
              />
            );
          })}
          <circle r={14} fill={C.green} />
        </g>
      </svg>
      <div style={{position: 'absolute', left: 1150, top: 300, width: 680, fontSize: 36, lineHeight: 1.6}}>
        <div style={{opacity: fadeIn(frame, tSparse)}}>
          h ≈ Σ c_i · v_i, &nbsp;c_i ≥ 0
        </div>
        <div style={{opacity: fadeIn(frame, tSparse + 20), color: C.amber}}>
          k ≈ 25 active tokens
        </div>
        <div style={{opacity: fadeIn(frame, tSmall), marginTop: 40, color: C.dim}}>share of activation variance</div>
        <div style={{opacity: fadeIn(frame, tSmall), display: 'flex', alignItems: 'center'}}>
          <div style={{width: 500, height: 30, border: `2px solid ${C.dim}`}}>
            <div style={{width: `${pct}%`, height: '100%', background: C.amber}} />
          </div>
          <span style={{marginLeft: 20, color: C.amber}}>&lt;10%</span>
        </div>
        <div style={{marginTop: 50, fontSize: 44, color: C.green, textShadow: glow(C.green, 16), opacity: s, transform: `translateY(${(1 - s) * 20}px)`}}>
          small · sparse · poised for speech
        </div>
      </div>
    </AbsoluteFill>
  );
};
