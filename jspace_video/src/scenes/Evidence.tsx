import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, Chip, fadeIn, glow, SectionTag, Typed, useSpring} from '../ui';
import {useCue} from '../cue';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// Two-hop reasoning: "spider" appears silently mid-network; swapping it changes the answer.
export const Spider: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tNever = cue('never writes');
  const tLights = cue('But spider');
  const tSwap = cue('Swap');
  const tCausal = cue('The hidden');
  const lit = useSpring(tLights, 14);
  const swap = useSpring(tSwap, 14);
  const layers = 12;
  const band = [4, 5, 6, 7, 8];
  return (
    <AbsoluteFill>
      <SectionTag n="04" label="A HIDDEN INTERMEDIATE STEP" />
      <div style={{position: 'absolute', top: 150, left: 120, fontSize: 40, color: C.white}}>
        <Typed text="Q: How many legs does the animal that spins webs have?" start={8} cps={34} />
      </div>
      {/* layer stack */}
      <div style={{position: 'absolute', left: 120, top: 300, display: 'flex', gap: 14, alignItems: 'flex-end'}}>
        {Array.from({length: layers}).map((_, l) => {
          const inBand = band.includes(l);
          const h = 360;
          return (
            <div key={l} style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
              <div
                style={{
                  width: 58,
                  height: h,
                  border: `2px solid ${inBand && frame > tLights ? C.amber : C.faint}`,
                  background: inBand && frame > tLights ? `rgba(255,176,0,${0.12 * lit})` : 'transparent',
                }}
              />
              <div style={{fontSize: 18, color: C.dim, marginTop: 6}}>L{l * 4}</div>
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 120, top: 740, fontSize: 24, color: C.dim}}>
        J-lens readout at the last question token, by layer
      </div>
      {/* the silent thought */}
      <div style={{position: 'absolute', left: 385, top: 390, transform: `scale(${lit})`, opacity: lit}}>
        <div style={{position: 'relative'}}>
          <Chip word="spider" style={{opacity: 1 - swap, position: 'absolute'}} scale={1.3} />
          <Chip word="ant" color={C.red} style={{opacity: swap, position: 'absolute', transform: `translateY(${(1 - swap) * -30}px)`}} scale={1.3} />
        </div>
      </div>
      <div style={{position: 'absolute', left: 120, top: 790, fontSize: 30, color: C.amber, opacity: fadeIn(frame, tSwap)}}>
        swap v_spider → v_ant
      </div>
      {/* output panel */}
      <div style={{position: 'absolute', left: 1200, top: 320, width: 600, border: `2px solid ${C.dim}`, padding: 30, fontSize: 36}}>
        <div style={{color: C.dim, fontSize: 24, marginBottom: 16}}>MODEL OUTPUT</div>
        <div style={{opacity: fadeIn(frame, tNever)}}>
          A: <span style={{color: swap > 0.5 ? C.red : C.green, textShadow: glow(swap > 0.5 ? C.red : C.green, 10)}}>{swap > 0.5 ? '6' : '8'}</span>
          <span style={{color: C.dim}}> legs</span>
        </div>
        <div style={{color: C.dim, fontSize: 24, marginTop: 20, opacity: fadeIn(frame, tNever + 10)}}>
          "spider" never appears in the text
        </div>
      </div>
      <div
        style={{
          position: 'absolute',
          left: 1200,
          top: 600,
          fontSize: 44,
          color: C.amber,
          textShadow: glow(C.amber, 16),
          opacity: fadeIn(frame, tCausal),
        }}
      >
        the hidden thought is causal
      </div>
      <div style={{position: 'absolute', left: 1200, top: 670, fontSize: 20, color: C.dim, opacity: fadeIn(frame, tSwap + 20)}}>
        (swap target shown is illustrative)
      </div>
    </AbsoluteFill>
  );
};

const PROPS = [
  {k: 'VERBAL REPORT', d: 'can say what is in it', cue: 'supports verbal'},
  {k: 'DIRECTED MODULATION', d: '"concentrate on citrus" → citrus appears', cue: 'steered'},
  {k: 'INTERNAL REASONING', d: 'carries intermediate sums: 21 → 42 → 49', cue: 'carries internal'},
  {k: 'FLEXIBLE GENERALIZATION', d: 'swap France→China: capital, language, currency follow', cue: 'generalizes'},
  {k: 'SELECTIVITY', d: 'small, privileged subset of processing', cue: 'selective'},
];

export const Five: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tPass = cue('passes all');
  return (
    <AbsoluteFill>
      <SectionTag n="05" label="FIVE MARKS OF A GLOBAL WORKSPACE" />
      <div style={{position: 'absolute', top: 170, left: 160, right: 160}}>
        {PROPS.map((p, i) => {
          const t = cue(p.cue);
          const o = fadeIn(frame, t);
          const check = frame > tPass + i * 5;
          return (
            <div
              key={p.k}
              style={{
                display: 'flex',
                alignItems: 'center',
                marginBottom: 26,
                opacity: o,
                transform: `translateX(${(1 - o) * -40}px)`,
                border: `2px solid ${check ? C.green : C.faint}`,
                padding: '20px 30px',
                background: check ? 'rgba(51,255,102,0.06)' : 'transparent',
              }}
            >
              <div style={{width: 60, fontSize: 40, color: C.green, textShadow: glow(C.green, 10)}}>{check ? '✓' : '·'}</div>
              <div style={{width: 560, fontSize: 38, color: C.green}}>{p.k}</div>
              <div style={{fontSize: 28, color: C.dim}}>{p.d}</div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const SURVIVES = ['fluent speech', 'sentiment classification', 'simple trivia / MMLU', 'continuing a language'];
const BREAKS = ['multi-hop reasoning', 'summarization', 'mental math', 'explicit report'];

export const Select: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tKnock = cue('Knock');
  const tBreak = cue('What breaks');
  const tAuto = cue('Automatic');
  return (
    <AbsoluteFill>
      <SectionTag n="06" label="SELECTIVITY: ABLATE THE WORKSPACE" />
      <div style={{position: 'absolute', top: 150, left: 0, right: 0, textAlign: 'center', fontSize: 34, color: C.red, opacity: fadeIn(frame, tKnock)}}>
        h ← h − Σ proj(h, v_top-k)
      </div>
      {[
        {title: 'STILL WORKS', items: SURVIVES, color: C.green, t: tKnock + 20, x: 200, sub: 'automatic'},
        {title: 'BREAKS', items: BREAKS, color: C.red, t: tBreak, x: 1020, sub: 'flexible / deliberate'},
      ].map((col) => (
        <div key={col.title} style={{position: 'absolute', left: col.x, top: 250, width: 700}}>
          <div style={{fontSize: 44, color: col.color, textShadow: glow(col.color, 12), opacity: fadeIn(frame, col.t)}}>{col.title}</div>
          <div style={{fontSize: 24, color: C.dim, marginBottom: 20, opacity: fadeIn(frame, col.t)}}>{col.sub}</div>
          {col.items.map((it, i) => {
            const o = fadeIn(frame, col.t + 10 + i * 8);
            const bar = col.color === C.red ? interpolate(frame, [col.t + 10 + i * 8, col.t + 40 + i * 8], [1, 0.25 + 0.1 * i], clamp) : 1;
            return (
              <div key={it} style={{opacity: o, marginBottom: 22}}>
                <div style={{fontSize: 30, color: C.white}}>{it}</div>
                <div style={{width: 560 * bar, height: 12, background: col.color, marginTop: 6, opacity: 0.8}} />
              </div>
            );
          })}
        </div>
      ))}
      <div style={{position: 'absolute', bottom: 150, left: 0, right: 0, textAlign: 'center', fontSize: 36, color: C.amber, opacity: fadeIn(frame, tAuto)}}>
        automatic runs underneath · deliberate runs through
      </div>
    </AbsoluteFill>
  );
};
