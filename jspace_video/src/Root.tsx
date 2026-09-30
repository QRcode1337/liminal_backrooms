import React from 'react';
import {Composition} from 'remotion';
import {FPS, JSpace, JSpaceVertical, totalFrames} from './Video';
import {JSpaceShort, SHORT_FPS, shortFrames} from './short/Short';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="JSpace" component={JSpace} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
    <Composition id="JSpaceVertical" component={JSpaceVertical} durationInFrames={totalFrames()} fps={FPS} width={1080} height={1920} />
    <Composition id="JSpaceShort" component={JSpaceShort} durationInFrames={shortFrames()} fps={SHORT_FPS} width={1080} height={1920} />
  </>
);
