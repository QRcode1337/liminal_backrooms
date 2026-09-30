import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, Series, staticFile} from 'remotion';
import narration from '../../scripts/short_narration.json';
import durations from '../short-durations.json';
import {CueProvider} from '../cue';
import {CRT, FontFace, Screen} from '../ui';
import {GlitchCut, KineticCaptions} from './fx';
import {Close, Hook, LensShort, Safety, Sfx, SpiderShort, Tiny} from './Scenes';

export const SHORT_FPS = 30;
const LEAD = 8;
// Held frames after each line: room for the punchline to land.
const TAIL: Record<string, number> = {hook: 40, lens: 36, spider: 70, tiny: 60, safety: 60, close: 185};
const SCENES: Record<string, React.FC> = {
  hook: Hook,
  lens: LensShort,
  spider: SpiderShort,
  tiny: Tiny,
  safety: Safety,
  close: Close,
};

const dur = durations as Record<string, number>;
const frames = (id: string) => LEAD + Math.ceil(dur[id] * SHORT_FPS) + TAIL[id];
export const shortFrames = () => narration.reduce((a, n) => a + frames(n.id), 0);
const closeStart = narration.slice(0, -1).reduce((a, n) => a + frames(n.id), 0);

// 1080x1920 native vertical cut (~75 s) for TikTok / Reels / Shorts.
export const JSpaceShort: React.FC = () => (
  <Screen>
    <FontFace />
    <Audio
      src={staticFile('beat.wav')}
      volume={(f) =>
        interpolate(f, [0, 20, closeStart, closeStart + 15, closeStart + 250, closeStart + 262], [0, 0.32, 0.32, 0.1, 0.1, 0.4], {
          extrapolateRight: 'clamp',
        })
      }
    />
    <Series>
      {narration.map((n) => {
        const Scene = SCENES[n.id];
        return (
          <Series.Sequence key={n.id} durationInFrames={frames(n.id)} premountFor={SHORT_FPS}>
            <CueProvider text={n.text} seconds={dur[n.id]} lead={LEAD}>
              <Scene />
              <KineticCaptions text={n.text} seconds={dur[n.id]} lead={LEAD} />
              <GlitchCut />
              <Sfx at={0} name="whoosh" volume={0.35} />
              <Sequence from={LEAD} layout="none">
                <Audio src={staticFile(`vo-short/${n.id}.wav`)} volume={1.1} />
              </Sequence>
            </CueProvider>
          </Series.Sequence>
        );
      })}
    </Series>
    <CRT />
  </Screen>
);
