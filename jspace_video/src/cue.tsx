import React, {createContext, useContext} from 'react';
import {useVideoConfig} from 'remotion';

type CueCtx = {text: string; seconds: number; lead: number};

const Ctx = createContext<CueCtx>({text: '', seconds: 1, lead: 0});

export const CueProvider: React.FC<CueCtx & {children: React.ReactNode}> = ({children, ...v}) => (
  <Ctx.Provider value={v}>{children}</Ctx.Provider>
);

// Returns the frame (scene-local) at which a phrase is roughly spoken,
// estimated from its character position in the narration.
export const useCue = () => {
  const {text, seconds, lead} = useContext(Ctx);
  const {fps} = useVideoConfig();
  return (phrase: string, offsetSec = 0) => {
    const i = text.indexOf(phrase);
    if (i < 0) throw new Error(`cue not found: ${phrase}`);
    return Math.round(lead + (i / text.length) * seconds * fps + offsetSec * fps);
  };
};
