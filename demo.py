import os
import warnings
import numpy as np
import torch
from io import StringIO

from src.app import LANGUAGES, WhisperTranscriber
from src.utils import slugify, write_srt, write_vtt
from src.whisperContainer import WhisperContainer

from translators.google import GoogleTranslator
from translators.papago import PapagoTranslator

import gradio as gr

warnings.filterwarnings(action='ignore')

language_dict = {'Korean': 'KOR',
                 'Japanese': 'JPN',
                 'English': 'ENG',
                 'Vietnamese': 'VIN',
                 'German': 'DEU',
                 'Italian': 'ITA',
                 'Portuguese': 'PTB',
                 'Russian': 'RUS',
                 'Spanish': 'ESP',
                 'French': 'FRA',
                 'Chinese': 'CHS',
                 'Hindi': 'HIN',
                 'Thai': 'THA',
                 'Indonesian': 'IDN'}

language_list = ['Korean', 'Japanese', 'English', 'Vietnamese', 'German', 'Italian', 
                 'Portuguese', 'Russian', 'Spanish', 'French', 'Chinese', 'Hindi', 
                 'Thai', 'Indonesian']
				 
args = {'best_of' : 5,
        'beam_size' : 5,
        'patience' : None,
        'length_penalty'  : None,
        'suppress_tokens' : "-1",
        'initial_prompt' : None,
        'condition_on_previous_text' : True,
        'fp16' : True,
        'compression_ratio_threshold' : 2.4,
        'logprob_threshold' : -1.0,
        'no_speech_threshold' : 0.6}

temperature = 0
temperature_increment_on_fallback = 0.2

vad = 'silero-vad'
vad_merge_window = 5
vad_max_merge_size = 30
vad_padding = 2
vad_prompt_window = 3
vad_cpu_cores = 1
auto_parallel = ''
task = 'transcribe'

if temperature_increment_on_fallback is not None:
    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
else:
    temperature = [temperature]
	
async def translate_caption(source_path,language,target_language,translator,model_name,file_format):
    model = WhisperContainer(model_name, device='cuda')
    transcriber = WhisperTranscriber(delete_uploaded_files=False, vad_cpu_cores=vad_cpu_cores)
    
    if translator == 'google':
        Translator = GoogleTranslator()
    else:
        Translator = PapagoTranslator()
        
    source_name = os.path.basename(source_path)
    file_path = '.'.join(source_path.split('.')[:-1]) + '.' + file_format
    
    result = transcriber.transcribe_file(model, 
                                     source_path,
                                     language = language,
                                     task = task,
                                     temperature=temperature, 
                                     vad=vad, 
                                     vadMergeWindow=vad_merge_window, 
                                     vadMaxMergeSize=vad_max_merge_size, 
                                     vadPadding=vad_padding, 
                                     vadPromptWindow=vad_prompt_window,
                                     **args)
    
    queries = [i['text'] for i in result['segments']]
    
    if language != target_language:
        queries = await Translator.translate(language_dict[language],
                                              language_dict[target_language],
                                              queries)

    for i in range(len(result['segments'])):
        result['segments'][i]['text'] = queries[i]

    if (target_language and target_language.lower() in ["Japanese","Chinese"]):
        maxLineWidth = 40
    else:
        maxLineWidth = 80

    segmentStream = StringIO()

    if file_format == 'vtt':
        write_vtt(result['segments'], file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        write_srt(result['segments'], file=segmentStream, maxLineWidth=maxLineWidth)

    segmentStream.seek(0)
    text = segmentStream.read()
    with open(file_path, 'w+', encoding="utf-8") as file:
        file.write(text)
    return text,file_path
	
demo = gr.Interface(fn=translate_caption,
                    inputs=[gr.Video(label='input'),
                            gr.Dropdown(language_list,value='Japanese',label= 'source language'),
                            gr.Dropdown(language_list,value='Korean',label= 'target language'),
                            gr.Dropdown(['google','papago'],value='papago',label= 'translator'),
                            gr.Dropdown(['tiny','base','small','medium','large','large-v2'],value='large-v2',label= 'model name'),
                            gr.Dropdown(['vtt','srt'],value='srt',label= 'file format')],
                    outputs=[gr.Text(label="output"),
                             gr.File(label="Download")])
demo.launch()