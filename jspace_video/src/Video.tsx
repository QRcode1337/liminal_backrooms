import React from 'react';
import {AbsoluteFill, Audio, Sequence, staticFile} from 'remotion';
import {linearTiming, TransitionSeries} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';
import narration from '../scripts/narration.json';
import durations from './vo-durations.json';
import {CueProvider} from './cue';
import {Captions, CRT, FontFace, Screen} from './ui';
import {Cold, Problem} from './scenes/Intro';
import {Define, Lens} from './scenes/Lens';
import {Five, Select, Spider} from './scenes/Evidence';
import {Audit, Caveat, Outro, Post, Structure} from './scenes/Deep';

export const FPS = 30;
const LEAD = 12; // frames of silence before narration starts in each scene
const TAIL = 24; // frames held after narration ends
const XFADE = 12;

const SCENES: Record<string, React.FC> = {
  cold: Cold,
  problem: Problem,
  lens: Lens,
  define: Define,
  spider: Spider,
  five: Five,
  select: Select,
  structure: Structure,
  audit: Audit,
  post: Post,
  caveat: Caveat,
  outro: Outro,
};

const dur = durations as Record<string, number>;

export const sceneFrames = (id: string) => LEAD + Math.ceil(dur[id] * FPS) + TAIL;

export const totalFrames = () =>
  narration.reduce((a, n) => a + sceneFrames(n.id), 0) - XFADE * (narration.length - 1);

export const JSpace: React.FC = () => (
  <Screen>
    <FontFace />
    <Audio src={staticFile('drone.wav')} volume={0.18} />
    <TransitionSeries>
      {narration.flatMap((n, i) => {
        const Scene = SCENES[n.id];
        const items = [
          <TransitionSeries.Sequence key={n.id} durationInFrames={sceneFrames(n.id)} premountFor={FPS}>
            <CueProvider text={n.text} seconds={dur[n.id]} lead={LEAD}>
              <AbsoluteFill>
                <Scene />
                <Captions text={n.text} seconds={dur[n.id]} lead={LEAD} />
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
    <CRT />
  </Screen>
);
