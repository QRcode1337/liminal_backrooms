import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT, glow, rand} from '../ui';
import {useCue} from '../cue';
import {Burst, Corridor, Flash, GlitchText, impulse, Shake, Slam} from './fx';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

export type SfxName = 'impact' | 'whoosh' | 'glitch' | 'riser' | 'tick';

export const Sfx: React.FC<{at: number; name: SfxName; volume?: number}> = ({at, name, volume = 0.6}) => (
  <Sequence from={Math.max(0, Math.round(at))} layout="none">
    <Audio src={staticFile(`sfx/${name}.wav`)} volume={volume} />
  </Sequence>
);

const Center: React.FC<{top: number; children: React.ReactNode}> = ({top, children}) => (
  <div style={{position: 'absolute', top, left: 0, right: 0, display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
    {children}
  </div>
);

const Label: React.FC<{children: React.ReactNode; color?: string; size?: number}> = ({children, color = C.dim, size = 34}) => (
  <div style={{fontFamily: FONT, fontSize: size, letterSpacing: size * 0.25, color}}>{children}</div>
);

// 1 · HOOK
export const Hook: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const h1 = cue('hidden space');
  const h2 = cue('inside Claude');
  const h3 = cue('thoughts wait');
  const hits = [h1, h2];
  const drift = ['leverage', 'spider', 'honestly', 'BUT', 'France', 'ethical', 'eight', 'fake', 'citrus', 'survival'];
  return (
    <AbsoluteFill>
      <Corridor speed={1.4} boost={[0, ...hits]} />
      {/* words flying out of the vanishing point toward camera */}
      {frame > h3 &&
        drift.map((w, i) => {
          const t = ((frame - h3) / 40 + i / drift.length) % 1;
          const a = rand(i * 3) * Math.PI * 2;
          const d = Math.pow(t, 2) * 900;
          return (
            <div
              key={w}
              style={{
                position: 'absolute',
                left: 540 + Math.cos(a) * d,
                top: 820 + Math.sin(a) * d * 1.3,
                transform: `translate(-50%, -50%) scale(${0.3 + t * 2.2})`,
                fontFamily: FONT,
                fontSize: 40,
                color: C.amber,
                opacity: Math.min(1, t * 3) * (1 - t),
                textShadow: glow(C.amber, 10),
              }}
            >
              {w}
            </div>
          );
        })}
      <Shake hits={hits}>
        <Center top={430}>
          <Slam at={h1}>
            <GlitchText text="A HIDDEN SPACE" size={96} hits={[h1]} />
          </Slam>
          <div style={{height: 30}} />
          <Slam at={h2}>
            <GlitchText text="INSIDE CLAUDE" size={96} color={C.white} hits={[h2]} />
          </Slam>
        </Center>
      </Shake>
      <Flash hits={hits} />
      <Sfx at={0} name="riser" volume={0.5} />
      <Sfx at={h1} name="impact" />
      <Sfx at={h2} name="impact" />
      <Sfx at={h3} name="whoosh" volume={0.4} />
    </AbsoluteFill>
  );
};

// 2 · LENS
export const LensShort: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const tName = cue('Jacobian');
  const tPoint = cue('Point it');
  const tShows = cue('shows the words');
  const tNow = cue('Now');
  const tLater = cue('or later');
  const layers = 10;
  const top = 380;
  const gap = 72;
  const scanStart = tPoint;
  const scanY = interpolate(frame, [scanStart, scanStart + 60], [top - 40, top + layers * gap], clamp);
  const hitsLayers = Array.from({length: layers}).map((_, i) => Math.round(scanStart + (i / layers) * 60));
  const vocab = [['spider', 'web', 'eight'], ['France', 'Paris'], ['sum', '42', '49'], ['citrus', 'lemon'], ['honestly', 'ethical'], ['BUT', 'sorry']];
  return (
    <AbsoluteFill>
      <Corridor speed={0.8} vpY={760} boost={[tName, tNow, tLater]} />
      <Shake hits={[tName, tNow, tLater]} amp={18}>
        <Center top={200}>
          <Slam at={tName}>
            <GlitchText text="JACOBIAN LENS" size={92} hits={[tName]} />
          </Slam>
        </Center>
        {Array.from({length: layers}).map((_, i) => {
          const lit = frame >= hitsLayers[i];
          const k = impulse(frame, [hitsLayers[i]], 18);
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: 150,
                right: 150,
                top: top + i * gap,
                height: 34,
                border: `2px solid ${lit ? C.green : C.faint}`,
                background: lit ? `rgba(51,255,102,${0.08 + 0.35 * k})` : 'rgba(0,0,0,0.4)',
                boxShadow: k > 0 ? `0 0 ${40 * k}px ${C.green}` : undefined,
                opacity: interpolate(frame, [i * 2, i * 2 + 8], [0, 1], clamp),
              }}
            />
          );
        })}
        {/* the lens beam */}
        {frame >= scanStart && (
          <div
            style={{
              position: 'absolute',
              left: 60,
              right: 60,
              top: scanY,
              height: 8,
              background: C.amber,
              boxShadow: `0 0 40px 12px ${C.amber}`,
              opacity: interpolate(frame, [scanStart + 60, scanStart + 75], [1, 0], clamp),
            }}
          />
        )}
        {hitsLayers.filter((_, i) => i % 2 === 1).map((h, j) => (
          <Burst key={h} at={Math.max(h, tShows - 30 + j * 6)} words={vocab[j % vocab.length]} x={540} y={top + (j * 2 + 1) * gap + 17} spread={380} />
        ))}
        <div style={{position: 'absolute', top: 1140, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 60}}>
          <Slam at={tNow}>
            <GlitchText text="NOW" size={84} color={C.amber} hits={[tNow]} />
          </Slam>
          <Slam at={tLater}>
            <GlitchText text="OR LATER" size={84} color={C.amber} hits={[tLater]} />
          </Slam>
        </div>
      </Shake>
      <Flash hits={[tName]} />
      <Sfx at={tName} name="impact" />
      <Sfx at={scanStart} name="riser" volume={0.35} />
      {hitsLayers.map((h) => (
        <Sfx key={h} at={h} name="tick" volume={0.25} />
      ))}
      <Sfx at={tNow} name="impact" volume={0.45} />
      <Sfx at={tLater} name="impact" volume={0.5} />
    </AbsoluteFill>
  );
};

// 3 · SPIDER
export const SpiderShort: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const tQ = cue('how many');
  const tNever = cue('never says');
  const tLights = cue('spider lights');
  const tSwap = cue('Swap');
  const tSix = cue('six');
  const q = 'HOW MANY LEGS DOES THE ANIMAL THAT SPINS WEBS HAVE?';
  const nq = Math.floor(interpolate(frame, [tQ, tQ + 55], [0, q.length], clamp));
  const swapped = frame >= tSwap + 6;
  const six = frame >= tSix;
  const ring = spring({frame: frame - tLights, fps, config: {damping: 12}});
  const hits = [tLights, tSwap + 6, tSix];
  return (
    <AbsoluteFill>
      <Corridor speed={0.6} vpY={900} color={swapped ? C.red : C.green} boost={hits} />
      <Shake hits={hits}>
        <div style={{position: 'absolute', top: 210, left: 90, right: 150, fontFamily: FONT, fontSize: 58, lineHeight: 1.25, color: C.white}}>
          {q.slice(0, nq)}
          {nq < q.length && nq > 0 && <span style={{color: C.green}}>▌</span>}
        </div>
        {/* hidden-thought ring */}
        <div
          style={{
            position: 'absolute',
            left: 540 - 280,
            top: 860 - 280,
            width: 560,
            height: 560,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <svg width={560} height={560} style={{position: 'absolute', inset: 0}}>
            {[0, 1, 2].map((i) => (
              <circle
                key={i}
                cx={280}
                cy={280}
                r={(120 + i * 50) * ring}
                fill="none"
                stroke={swapped ? C.red : C.amber}
                strokeOpacity={0.6 - i * 0.18}
                strokeWidth={3}
                strokeDasharray="10 14"
                transform={`rotate(${frame * (i % 2 ? -2 : 2)} 280 280)`}
              />
            ))}
          </svg>
          {frame >= tLights && (
            <div style={{transform: `scale(${ring})`}}>
              <GlitchText
                text={swapped ? 'ANT' : 'SPIDER'}
                size={120}
                color={swapped ? C.red : C.amber}
                hits={[tLights, tSwap + 6]}
              />
            </div>
          )}
        </div>
        {frame >= tLights && (
          <Center top={1110}>
            <Label color={C.amber}>INSIDE THE MODEL · UNSPOKEN</Label>
          </Center>
        )}
        {/* answer */}
        <div
          style={{
            position: 'absolute',
            top: 470,
            left: 90,
            display: 'flex',
            alignItems: 'center',
            gap: 30,
            fontFamily: FONT,
            opacity: interpolate(frame, [tNever - 10, tNever], [0, 1], clamp),
          }}
        >
          <Label>ANSWER →</Label>
          <div style={{fontSize: 120, lineHeight: 1, color: six ? C.red : C.green, textShadow: glow(six ? C.red : C.green, 30)}}>
            <Slam at={six ? tSix : tNever - 10} from={six ? 2.2 : 1.4}>
              {six ? '6' : '8'}
            </Slam>
          </div>
        </div>
        {frame >= tSwap && (
          <div style={{position: 'absolute', top: 1175, left: 0, right: 0, textAlign: 'center', fontFamily: FONT, fontSize: 26, color: C.dim}}>
            illustrative swap · arXiv:2607.15495
          </div>
        )}
      </Shake>
      <Flash hits={[tLights]} color={C.amber} />
      <Flash hits={[tSix]} color={C.red} strength={0.6} />
      {Array.from({length: 8}).map((_, i) => (
        <Sfx key={i} at={tQ + i * 7} name="tick" volume={0.18} />
      ))}
      <Sfx at={tLights} name="impact" />
      <Sfx at={tLights} name="glitch" volume={0.4} />
      <Sfx at={tSwap + 6} name="glitch" volume={0.55} />
      <Sfx at={tSix} name="impact" volume={0.75} />
    </AbsoluteFill>
  );
};

// 4 · TINY
export const Tiny: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const t25 = cue('twenty-five');
  const t10 = cue('Under ten');
  const tKnock = cue('Knock');
  const tTalks = cue('still talks');
  const tReason = cue("can't reason");
  const knocked = frame >= tKnock;
  const count = Math.min(25, Math.max(0, Math.floor((frame - t25 + 20) / 1.4)));
  const pct = interpolate(frame, [t10, t10 + 25], [0, 9.6], clamp);
  const hits = [tKnock, tReason];
  return (
    <AbsoluteFill>
      <Corridor speed={knocked ? 0.25 : 1} color={knocked ? C.red : C.green} boost={[tKnock]} />
      <Shake hits={hits} amp={34}>
        <Center top={200}>
          <GlitchText text={`${count}`} size={200} hits={count === 25 ? [Math.round(t25 + 15)] : []} color={knocked ? C.red : C.green} />
          <Label color={C.white} size={36}>CONCEPTS AT ONCE</Label>
        </Center>
        {/* 5x5 slots */}
        <div style={{position: 'absolute', top: 560, left: 540 - 250, width: 500, height: 500}}>
          {Array.from({length: 25}).map((_, i) => {
            const on = i < count;
            const dx = knocked ? (rand(i) - 0.5) * 1400 * interpolate(frame, [tKnock, tKnock + 40], [0, 1], clamp) : 0;
            const dy = knocked ? (rand(i + 50) - 0.3) * 1600 * Math.pow(interpolate(frame, [tKnock, tKnock + 40], [0, 1], clamp), 1.5) : 0;
            const rot = knocked ? (rand(i + 9) - 0.5) * 720 * interpolate(frame, [tKnock, tKnock + 40], [0, 1], clamp) : 0;
            return (
              <div
                key={i}
                style={{
                  position: 'absolute',
                  left: (i % 5) * 100 + 10,
                  top: Math.floor(i / 5) * 100 + 10,
                  width: 80,
                  height: 80,
                  border: `3px solid ${knocked ? C.red : C.green}`,
                  background: on ? (knocked ? C.red : C.green) : 'transparent',
                  boxShadow: on ? `0 0 18px ${knocked ? C.red : C.green}` : undefined,
                  transform: `translate(${dx}px, ${dy}px) rotate(${rot}deg)`,
                  opacity: knocked ? interpolate(frame, [tKnock + 20, tKnock + 40], [1, 0], clamp) : 1,
                }}
              />
            );
          })}
        </div>
        {!knocked && frame >= t10 && (
          <div style={{position: 'absolute', top: 1110, left: 190, right: 190}}>
            <div style={{height: 40, border: `3px solid ${C.green}`}}>
              <div style={{width: `${pct * 10}%`, height: '100%', background: C.amber, boxShadow: `0 0 20px ${C.amber}`}} />
            </div>
            <div style={{fontFamily: FONT, fontSize: 40, color: C.amber, marginTop: 10, textAlign: 'center'}}>&lt;10% OF THE SIGNAL</div>
          </div>
        )}
        {knocked && (
          <Center top={650}>
            <Slam at={tTalks}>
              <GlitchText text="✓ STILL TALKS" size={84} color={C.green} hits={[tTalks]} />
            </Slam>
            <div style={{height: 60}} />
            <Slam at={tReason}>
              <GlitchText text="✗ CAN'T REASON" size={92} color={C.red} hits={[tReason]} />
            </Slam>
          </Center>
        )}
      </Shake>
      <Flash hits={[tKnock]} color={C.red} strength={0.7} />
      {Array.from({length: 13}).map((_, i) => (
        <Sfx key={i} at={t25 - 20 + i * 2.8} name="tick" volume={0.2} />
      ))}
      <Sfx at={tKnock} name="impact" volume={0.8} />
      <Sfx at={tKnock} name="glitch" volume={0.5} />
      <Sfx at={tTalks} name="whoosh" volume={0.35} />
      <Sfx at={tReason} name="impact" volume={0.7} />
    </AbsoluteFill>
  );
};

// 5 · SAFETY
export const Safety: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const tTest = cue('staged blackmail');
  const words: [string, number, string][] = [
    ['LEVERAGE', cue('leverage'), C.amber],
    ['SURVIVAL', cue('Survival'), C.amber],
    ['FAKE', cue('Fake'), C.green],
    ['SECRETLY', cue('Secretly'), C.red],
    ['FRAUD', cue('Fraud'), C.red],
  ];
  const tSuspect = cue('suspected');
  const tCheat = cue('trained to cheat');
  const tClean = cue('While its');
  const hits = words.map((w) => w[1]);
  const active = words.filter((w) => frame >= w[1]);
  const current = active[active.length - 1];
  const alarm = frame >= tTest ? 0.5 + 0.5 * Math.sin(frame / 4) : 0;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{background: `radial-gradient(circle at 50% 45%, rgba(255,77,77,${0.12 * alarm}), transparent 70%)`}} />
      <Corridor speed={1.2} color={frame >= tCheat ? C.red : C.amber} boost={hits} />
      <Shake hits={hits} amp={30}>
        <Center top={200}>
          <Label color={C.red} size={38}>
            {frame >= tCheat ? '● MODEL TRAINED TO CHEAT' : '● STAGED BLACKMAIL TEST'}
          </Label>
          <div style={{fontFamily: FONT, fontSize: 28, color: C.dim, marginTop: 10}}>J-LENS · WHAT IT DIDN'T SAY</div>
        </Center>
        {/* pile of previous words */}
        <div style={{position: 'absolute', top: 1010, left: 80, right: 150, display: 'flex', flexWrap: 'wrap', gap: 16, justifyContent: 'center'}}>
          {active.slice(0, -1).map(([w, , col]) => (
            <span key={w} style={{fontFamily: FONT, fontSize: 40, color: col, border: `2px solid ${col}`, padding: '4px 14px', textShadow: glow(col, 8)}}>
              {w}
            </span>
          ))}
        </div>
        {current && (
          <Center top={560}>
            <Slam key={current[0]} at={current[1]} from={4}>
              <GlitchText text={current[0]} size={150} color={current[2]} hits={[current[1]]} />
            </Slam>
            {current[0] === 'FAKE' && frame >= tSuspect && (
              <div style={{fontFamily: FONT, fontSize: 44, color: C.white, marginTop: 40}}>it suspected a test</div>
            )}
          </Center>
        )}
        {frame >= tClean && (
          <div
            style={{
              position: 'absolute',
              top: 820,
              left: 180,
              right: 180,
              border: `2px solid ${C.green}`,
              background: 'rgba(0,0,0,0.75)',
              padding: 24,
              fontFamily: FONT,
              fontSize: 34,
              color: C.green,
            }}
          >
            <div>return sorted(xs)</div>
            <div style={{color: C.white}}>✓ output looks clean</div>
          </div>
        )}
      </Shake>
      <Flash hits={hits.slice(0, 3)} color={C.amber} strength={0.35} />
      <Flash hits={hits.slice(3)} color={C.red} strength={0.55} />
      <Sfx at={tTest} name="riser" volume={0.4} />
      {hits.map((h) => (
        <Sfx key={h} at={h} name="impact" volume={0.7} />
      ))}
      <Sfx at={cue('Fake')} name="glitch" volume={0.5} />
      <Sfx at={tCheat} name="whoosh" volume={0.4} />
    </AbsoluteFill>
  );
};

// 6 · CLOSE
export const Close: React.FC = () => {
  const cue = useCue();
  const frame = useCurrentFrame();
  const tNot = cue("It's not");
  const tFirst = cue('for the first');
  const tRead = cue('we can read');
  const tJ = cue('This is');
  const bigHit = tJ + 14;
  return (
    <AbsoluteFill>
      <Corridor speed={frame >= bigHit ? 2.2 : 0.35} boost={[bigHit]} />
      <Shake hits={[bigHit]} amp={40}>
        {frame < bigHit && (
          <Center top={430}>
            <div style={{fontFamily: FONT, fontSize: 56, color: C.dim, opacity: interpolate(frame, [tNot, tNot + 10], [0, 1], clamp)}}>
              NOT PROOF OF CONSCIOUSNESS.
            </div>
            <div style={{height: 50}} />
            <Slam at={tFirst} from={1.8}>
              <div style={{fontFamily: FONT, fontSize: 64, color: C.white, textAlign: 'center', padding: '0 80px'}}>BUT FOR THE FIRST TIME</div>
            </Slam>
            <div style={{height: 30}} />
            <Slam at={tRead} from={2.2}>
              <GlitchText text="WE CAN READ IT" size={96} hits={[tRead]} />
            </Slam>
          </Center>
        )}
        {frame >= bigHit && (
          <Center top={560}>
            <Slam at={bigHit} from={5}>
              <GlitchText text="J-SPACE" size={210} hits={[bigHit, bigHit + 30]} />
            </Slam>
            <div style={{fontFamily: FONT, fontSize: 34, marginTop: 40, opacity: interpolate(frame, [bigHit + 20, bigHit + 35], [0, 1], clamp), textAlign: 'center', lineHeight: 1.5, background: 'rgba(0,0,0,0.8)', padding: '16px 28px', color: C.green}}>
              Gurnee, Sofroniew, Lindsey et al.
              <br />
              Anthropic 2026 · arXiv:2607.15495
            </div>
          </Center>
        )}
      </Shake>
      <Flash hits={[bigHit]} strength={0.8} />
      <Sfx at={tRead - 45} name="riser" volume={0.5} />
      <Sfx at={tRead} name="impact" volume={0.5} />
      <Sfx at={bigHit} name="impact" volume={0.9} />
      <Sfx at={bigHit} name="glitch" volume={0.5} />
    </AbsoluteFill>
  );
};
