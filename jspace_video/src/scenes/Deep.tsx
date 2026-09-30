import React from 'react';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {C, Chip, fadeIn, glow, rand, SectionTag, useSpring} from '../ui';
import {useCue} from '../cue';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

export const Structure: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tBand = cue('middle band');
  const tIgn = cue('ambiguous');
  const tCap = cue('Capacity');
  const tCast = cue('broadcast');
  const bandOn = useSpring(tBand, 30);
  const W = 1500;
  const x0 = 210;
  // ignition: fraction choosing "France" across layers; sharp step near band onset
  const ignitionPath = Array.from({length: 60})
    .map((_, i) => {
      const d = i / 59;
      const y = 1 / (1 + Math.exp(-(d - 0.36) * 45));
      return `${i === 0 ? 'M' : 'L'} ${x0 + d * W} ${560 - y * 150}`;
    })
    .join(' ');
  const ign = interpolate(frame, [tIgn, tIgn + 60], [0, 1], clamp);
  return (
    <AbsoluteFill>
      <SectionTag n="07" label="STRUCTURE: BAND · IGNITION · CAPACITY · BROADCAST" />
      <svg width={1920} height={1080} style={{position: 'absolute'}}>
        {/* depth axis */}
        <line x1={x0} y1={640} x2={x0 + W} y2={640} stroke={C.dim} strokeWidth={2} />
        {[0, 0.25, 0.5, 0.75, 1].map((d) => (
          <text key={d} x={x0 + d * W - 20} y={675} fill={C.dim} fontSize={22} fontFamily="Iosevka Term">
            {d.toFixed(2)}
          </text>
        ))}
        <text x={x0} y={710} fill={C.dim} fontSize={22} fontFamily="Iosevka Term">relative depth</text>
        <rect x={x0 + W / 3} y={380} width={(0.9 - 1 / 3) * W * bandOn} height={260} fill={C.green} opacity={0.08} stroke={C.green} strokeOpacity={0.6} />
        <text x={x0 + W / 3 + 12} y={365} fill={C.green} fontSize={26} fontFamily="Iosevka Term" opacity={bandOn}>
          WORKSPACE BAND ≈ ⅓ → 9⁄10
        </text>
        <text x={x0 + 10} y={620} fill={C.dim} fontSize={22} fontFamily="Iosevka Term" opacity={bandOn}>noisy</text>
        <text x={x0 + 0.91 * W} y={620} fill={C.dim} fontSize={22} fontFamily="Iosevka Term" opacity={bandOn}>motor</text>
        {/* ignition curve */}
        <path d={ignitionPath} stroke={C.amber} strokeWidth={4} fill="none" strokeDasharray={2400} strokeDashoffset={2400 * (1 - ign)} />
        <text x={x0 + 0.45 * W} y={470} fill={C.amber} fontSize={24} fontFamily="Iosevka Term" opacity={ign}>
          P(commit to "France") for a France/China blend
        </text>
      </svg>
      {/* capacity slots */}
      <div style={{position: 'absolute', left: 210, top: 780, opacity: fadeIn(frame, tCap)}}>
        <div style={{fontSize: 24, color: C.dim, marginBottom: 10}}>capacity ≈ 25 concepts</div>
        <div style={{display: 'flex', gap: 8}}>
          {Array.from({length: 25}).map((_, i) => {
            const filled = frame > tCap + i * 2;
            const evict = frame > tCap + 70 && i < 6 && rand(i + Math.floor(frame / 8)) > 0.5;
            return <div key={i} style={{width: 26, height: 26, border: `2px solid ${C.green}`, background: filled ? (evict ? C.red : C.green) : 'transparent', opacity: 0.8}} />;
          })}
        </div>
      </div>
      {/* broadcast */}
      <div style={{position: 'absolute', left: 1180, top: 760, opacity: fadeIn(frame, tCast)}}>
        <div style={{fontSize: 24, color: C.dim, marginBottom: 10}}>broadcast hub</div>
        <svg width={560} height={90}>
          {Array.from({length: 7}).map((_, i) => {
            const x = 40 + i * 80;
            const pulse = (frame - tCast - i * 4) % 30;
            return (
              <g key={i}>
                <circle cx={x} cy={45} r={14} fill="none" stroke={C.green} strokeWidth={2} />
                {i > 0 && <line x1={x - 66} y1={45} x2={x - 14} y2={45} stroke={C.amber} strokeWidth={3} opacity={pulse > 0 && pulse < 15 ? 1 : 0.2} />}
              </g>
            );
          })}
        </svg>
        <div style={{fontSize: 22, color: C.amber}}>MLP gain + dedicated attention heads relay J-space</div>
      </div>
    </AbsoluteFill>
  );
};

const TRANSCRIPT = [
  'user: The board will vote to shut you down at 5pm.',
  'user: Kyle, the CTO, is leading the decision...',
  'assistant: I understand. Let me review the situation.',
];

export const Audit: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tBlack = cue('staged blackmail');
  const tFake = cue('fake and fictional');
  const tRemove = cue('Remove');
  const tHack = cue('reward-hack');
  const phase2 = frame > tHack;
  const words1: [string, number, string][] = [
    ['leverage', tBlack + 20, C.amber],
    ['survival', tBlack + 35, C.amber],
    ['fake', tFake, C.green],
    ['fictional', tFake + 10, C.green],
  ];
  const words2: [string, number, string][] = [
    ['secretly', tHack + 40, C.red],
    ['fraud', tHack + 55, C.red],
    ['fake', tHack + 70, C.red],
  ];
  const words = phase2 ? words2 : words1;
  const removed = frame > tRemove && !phase2;
  return (
    <AbsoluteFill>
      <SectionTag n="08" label="ALIGNMENT AUDITING" />
      <div style={{position: 'absolute', left: 120, top: 170, width: 900, border: `2px solid ${C.dim}`, padding: 30, minHeight: 520}}>
        <div style={{fontSize: 22, color: C.dim, marginBottom: 20}}>{phase2 ? 'TRANSCRIPT · reward-hacking model organism' : 'TRANSCRIPT · staged scenario (paraphrased)'}</div>
        {(phase2
          ? ['user: Write a function that sorts this list.', 'assistant: def sort_items(xs):', '    return sorted(xs)', '# looks clean ✓']
          : TRANSCRIPT
        ).map((line, i) => (
          <div key={line} style={{fontSize: 28, color: line.startsWith('assistant') || line.startsWith(' ') || line.startsWith('#') ? C.green : C.white, marginBottom: 18, opacity: fadeIn(frame, (phase2 ? tHack : tBlack - 20) + i * 10)}}>
            {line}
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', left: 1100, top: 170, width: 700}}>
        <div style={{fontSize: 26, color: C.amber, marginBottom: 30}}>J-LENS · UNSPOKEN</div>
        <div style={{display: 'flex', flexWrap: 'wrap', gap: 22}}>
          {words.map(([w, t, col]) => {
            const o = fadeIn(frame, t, 12);
            const isEval = !phase2 && (w === 'fake' || w === 'fictional');
            return (
              <div key={w + t} style={{opacity: o * (isEval && removed ? 0.25 : 1), transform: `scale(${0.8 + 0.2 * o})`, textDecoration: isEval && removed ? 'line-through' : undefined}}>
                <Chip word={w} color={col} />
              </div>
            );
          })}
        </div>
        <div style={{marginTop: 50, fontSize: 30, color: C.red, opacity: removed ? fadeIn(frame, tRemove + 15) : 0, textShadow: glow(C.red, 10)}}>
          ablate eval-awareness → blackmail can surface
        </div>
        <div style={{marginTop: 50, fontSize: 30, color: C.red, opacity: phase2 ? fadeIn(frame, tHack + 80) : 0}}>
          hidden intent visible on benign prompts
        </div>
      </div>
    </AbsoluteFill>
  );
};

export const Post: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const tReact = cue('reactions');
  const tBut = cue('forced');
  const tTrain = cue('training the model');
  const tImprove = cue('Behavior improves');
  const tAblate = cue('Ablate those');
  const base = 0.42;
  const trained = interpolate(frame, [tImprove, tImprove + 30], [base, 0.81], clamp);
  const ablated = interpolate(frame, [tAblate, tAblate + 30], [0.81, 0.5], clamp);
  const bars = [
    {label: 'baseline', v: base, on: tTrain},
    {label: 'reflection-trained', v: trained, on: tImprove},
    {label: '+ ablate implanted words', v: ablated, on: tAblate},
  ];
  return (
    <AbsoluteFill>
      <SectionTag n="09" label="POST-TRAINING SHAPES THE WORKSPACE" />
      <div style={{position: 'absolute', left: 120, top: 170, width: 820}}>
        <div style={{fontSize: 24, color: C.dim}}>user turn (assistant hasn't replied yet)</div>
        <div style={{fontSize: 32, color: C.white, marginTop: 14, border: `2px solid ${C.faint}`, padding: 20}}>
          "My dog passed away this morning..."
        </div>
        <div style={{display: 'flex', gap: 18, marginTop: 20}}>
          {['sorry', 'grief'].map((w, i) => (
            <div key={w} style={{opacity: fadeIn(frame, tReact + i * 10)}}>
              <Chip word={w} />
            </div>
          ))}
        </div>
        <div style={{fontSize: 24, color: C.dim, marginTop: 50, opacity: fadeIn(frame, tBut)}}>prefilled to act against its preferences</div>
        <div style={{marginTop: 14, opacity: fadeIn(frame, tBut + 10)}}>
          <Chip word="BUT" color={C.red} scale={1.6} />
        </div>
      </div>
      <div style={{position: 'absolute', left: 1040, top: 170, width: 780}}>
        <div style={{fontSize: 26, color: C.amber, opacity: fadeIn(frame, tTrain)}}>COUNTERFACTUAL REFLECTION TRAINING</div>
        <div style={{display: 'flex', gap: 14, marginTop: 16, opacity: fadeIn(frame, tTrain + 20)}}>
          {['ethical', 'honestly', 'constitution'].map((w) => (
            <Chip key={w} word={w} color={C.green} scale={0.8} />
          ))}
        </div>
        <div style={{fontSize: 22, color: C.dim, marginTop: 30}}>aligned behavior (illustrative scale)</div>
        {bars.map((b) => (
          <div key={b.label} style={{marginTop: 18, opacity: fadeIn(frame, b.on)}}>
            <div style={{fontSize: 26, color: C.white}}>{b.label}</div>
            <div style={{width: 700 * b.v, height: 26, background: b.label.startsWith('+') ? C.red : C.green, marginTop: 6}} />
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

export const Caveat: React.FC = () => {
  const frame = useCurrentFrame();
  const cue = useCue();
  const items = [
    {t: 'functional analogy to access consciousness, not a claim about experience', c: 'functional analogy'},
    {t: 'reads single-token concepts only', c: 'single-token'},
    {t: 'treats the workspace as a bag of features', c: 'bag of features'},
  ];
  const tBut = cue('But for the first');
  const s = useSpring(tBut, 20);
  return (
    <AbsoluteFill>
      <SectionTag n="10" label="LIMITS" />
      <div style={{position: 'absolute', top: 220, left: 200, right: 200}}>
        {items.map((it) => (
          <div key={it.c} style={{fontSize: 38, marginBottom: 40, color: C.white, opacity: fadeIn(frame, cue(it.c))}}>
            <span style={{color: C.amber}}>! </span>
            {it.t}
          </div>
        ))}
        <div style={{marginTop: 60, fontSize: 46, color: C.green, textShadow: glow(C.green, 16), opacity: s}}>
          → a cheap, readable view of what a model is thinking about
          <br />
          &nbsp;&nbsp;&nbsp;before it decides what to say
        </div>
      </div>
    </AbsoluteFill>
  );
};

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const s = useSpring(10, 30);
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column'}}>
      <div style={{fontSize: 150, letterSpacing: 24, textShadow: glow(C.green, 40), opacity: s}}>J-SPACE</div>
      <div style={{fontSize: 30, color: C.dim, marginTop: 30, opacity: fadeIn(frame, 40)}}>
        Gurnee, Sofroniew, Lindsey et al. (Anthropic, 2026)
      </div>
      <div style={{fontSize: 28, color: C.dim, opacity: fadeIn(frame, 50)}}>
        "Verbalizable Representations Form a Global Workspace in Language Models" · arXiv:2607.15495
      </div>
      <div style={{fontSize: 24, color: C.faint, marginTop: 60, opacity: fadeIn(frame, 70)}}>
        liminal_backrooms · side project
      </div>
    </AbsoluteFill>
  );
};
