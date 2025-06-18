const AudioPlayer = {
  playBase64: (base64: string) => {
    const audio = new Audio(`data:audio/wav;base64,${base64}`);
    audio.play();
  },
};
export default AudioPlayer;
