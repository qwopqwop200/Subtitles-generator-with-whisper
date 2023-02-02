# whisper-and-nmt
Subtitles generator using whisper and translator
# conda environment setup and run
```
conda create --name whisper python=3.8 -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

python demo.py
```
## Acknowledgments
Many thanks to these excellent opensource projects
* [Whisper](https://github.com/openai/whisper)
* [Whisper Webui](https://huggingface.co/spaces/aadnk/whisper-webui)
* [Silero VAD](https://github.com/snakers4/silero-vad)
* [manga-image-translator](https://github.com/zyddnys/manga-image-translator)
