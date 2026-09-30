import React from 'react';
import {Composition} from 'remotion';
import {FPS, JSpace, totalFrames} from './Video';

export const RemotionRoot: React.FC = () => (
  <Composition id="JSpace" component={JSpace} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
);
