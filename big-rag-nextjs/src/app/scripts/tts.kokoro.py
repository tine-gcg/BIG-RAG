import sys
import soundfile as sf
import numpy as np
from kokoro import KPipeline

text, output = sys.argv[1], sys.argv[2]
pipeline = KPipeline(lang_code='a')
segments = [audio for _, _, audio in pipeline(text, voice='af_heart', speed=1, split_pattern=r'\n+')]
sf.write(output, np.concatenate(segments), samplerate=24000)
