# Idea:
- fine tune MusicGen with webscrapped audio
- reimplement DITTO to allow for inference-time control of music generation


## MusicGen
ai.honu.io/papers/musicgen
uses EnCodec 


Note: set the following before `pip install -U audiogen`:

```
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
```


pyenv install -s 3.9.17 && pyenv virtualenv 3.9.17 .env

arch -x86_64 pyenv install 3.9.13

# Fixing xformers compile error

      clang++: error: unsupported option '-fopenmp'
      error: command '/usr/bin/clang++' failed with exit code 1

brew install llvm libomp

export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++

pip install -e .
clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp main.cpp -o main

from audiocraft.models import MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-melody')  # or 'facebook/musicgen-medium'
model.set_generation_params(duration=8, use_sampling=True, top_k=250, cfg_coef=3.0)
descriptions = ['happy rock', 'ambient piano texture']
wav = model.generate(descriptions)  # returns tensor [B, C, T] with model.sample_rate

import torchaudio
melody, sr = torchaudio.load('my_melody.wav')  # [C, T]
wav = model.generate_with_chroma(['piano ballad'], [melody], sr)

prompt_wav, sr = torchaudio.load('prompt.wav')
wav = model.generate_continuation(prompt_wav, prompt_sample_rate=sr, descriptions=['continue: ambient pad'])

wav = model.generate_unconditional(3)  # 3 samples

m2 chip

