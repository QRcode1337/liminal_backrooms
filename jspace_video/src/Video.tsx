import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {linearTiming, TransitionSeries} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import narration from '../scripts/narration.json';
import durations from './vo-durations.json';
import {CueProvider} from './cue';
import {C, Captions, CRT, FontFace, glow, Screen, VerticalCtx} from './ui';
import {Cold, Problem} from './scenes/Intro';
import {Define, Lens} from './scenes/Lens';
import {Five, Select, Spider} from './scenes/Evidence';
import {Audit, Caveat, Outro, Post, Structure} from './scenes/Deep';

export const FPS = 30;
const LEAD = 12; // frames of silence before narration starts in each scene
const TAIL = 24; // frames held after narration ends
const XFADE = 12;

const SCENES: Record<string, {C: React.FC; tag?: string}> = {
  cold: {C: Cold},
  problem: {C: Problem, tag: '01 · THE QUESTION'},
  lens: {C: Lens, tag: '02 · THE JACOBIAN LENS'},
  define: {C: Define, tag: '03 · DEFINING J-SPACE'},
  spider: {C: Spider, tag: '04 · A HIDDEN STEP'},
  five: {C: Five, tag: '05 · FIVE MARKS'},
  select: {C: Select, tag: '06 · SELECTIVITY'},
  structure: {C: Structure, tag: '07 · STRUCTURE'},
  audit: {C: Audit, tag: '08 · AUDITING'},
  post: {C: Post, tag: '09 · POST-TRAINING'},
  caveat: {C: Caveat, tag: '10 · LIMITS'},
  outro: {C: Outro},
};

const dur = durations as Record<string, number>;

export const sceneFrames = (id: string) => LEAD + Math.ceil(dur[id] * FPS) + TAIL;

// Scene start frames on the global timeline (transitions overlap by XFADE).
const starts = narration.map((_, i) =>
  narration.slice(0, i).reduce((a, n) => a + sceneFrames(n.id) - XFADE, 0),
);

export const totalFrames = () =>
  narration.reduce((a, n) => a + sceneFrames(n.id), 0) - XFADE * (narration.length - 1);

// All scenes with narration audio; captions optional so layouts can place their own.
const SceneStack: React.FC<{captions: boolean}> = ({captions}) => (
  <TransitionSeries>
    {narration.flatMap((n, i) => {
      const Scene = SCENES[n.id].C;
      const items = [
        <TransitionSeries.Sequence key={n.id} durationInFrames={sceneFrames(n.id)} premountFor={FPS}>
          <CueProvider text={n.text} seconds={dur[n.id]} lead={LEAD}>
            <AbsoluteFill>
              <Scene />
              {captions && <Captions text={n.text} seconds={dur[n.id]} lead={LEAD} />}
            </AbsoluteFill>
            <Sequence from={LEAD} layout="none">
              <Audio src={staticFile(`vo/${n.id}.wav`)} />
            </Sequence>
          </CueProvider>
        </TransitionSeries.Sequence>,
      ];
      if (i > 0) {
        items.unshift(
          <TransitionSeries.Transition
            key={`${n.id}-t`}
            presentation={fade()}
            timing={linearTiming({durationInFrames: XFADE})}
          />,
        );
      }
      return items;
    })}
  </TransitionSeries>
);

export const JSpace: React.FC = () => (
  <Screen>
    <FontFace />
    <Audio src={staticFile('drone.wav')} volume={0.18} />
    <SceneStack captions />
    <CRT />
  </Screen>
);

// 1080x1920 (TikTok / Reels / Shorts). The 16:9 scenes are scaled into a middle
// band; chapter header above, large chunked captions below, kept clear of the
// platform UI (top ~150px, bottom ~400px, right ~140px).
const SCALE = 0.6;
const BAND_TOP = 470;
const BAND_H = 1080 * SCALE;

const VerticalHeader: React.FC<{tag?: string}> = ({tag}) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 12], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <div style={{position: 'absolute', top: 220, left: 70, right: 70, opacity: o}}>
      <div style={{fontSize: 34, letterSpacing: 10, color: C.dim}}>J-SPACE</div>
      {tag && (
        <div style={{fontSize: 56, color: C.green, textShadow: glow(C.green, 14), marginTop: 12}}>
          {tag}
        </div>
      )}
    </div>
  );
};

const Progress: React.FC = () => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  return (
    <div style={{position: 'absolute', top: BAND_TOP + BAND_H + 18, left: 0, right: 0, height: 6, background: C.faint}}>
      <div style={{width: `${(frame / durationInFrames) * 100}%`, height: '100%', background: C.green, boxShadow: `0 0 10px ${C.green}`}} />
    </div>
  );
};

export const JSpaceVertical: React.FC = () => (
  <Screen>
    <FontFace />
    <Audio src={staticFile('drone.wav')} volume={0.18} />
    <div
      style={{
        position: 'absolute',
        top: BAND_TOP,
        left: (1080 - 1920 * SCALE) / 2,
        width: 1920,
        height: 1080,
        transform: `scale(${SCALE})`,
        transformOrigin: 'top left',
        overflow: 'hidden',
      }}
    >
      <VerticalCtx.Provider value>
        <SceneStack captions={false} />
      </VerticalCtx.Provider>
    </div>
    <div style={{position: 'absolute', top: BAND_TOP - 2, left: 0, right: 0, height: 2, background: C.dim}} />
    <div style={{position: 'absolute', top: BAND_TOP + BAND_H, left: 0, right: 0, height: 2, background: C.dim}} />
    <Progress />
    {narration.map((n, i) => (
      <Sequence key={n.id} from={starts[i]} durationInFrames={sceneFrames(n.id)} layout="none">
        <VerticalHeader tag={SCENES[n.id].tag} />
        <Captions
          text={n.text}
          seconds={dur[n.id]}
          lead={LEAD}
          maxWords={6}
          style={{top: BAND_TOP + BAND_H + 70, bottom: 'auto', left: 70, right: 140, fontSize: 60, lineHeight: 1.3}}
        />
      </Sequence>
    ))}
    <CRT />
  </Screen>
);
