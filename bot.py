import sys
import signal
import discord
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import yt_dlp
from audio_separator.separator import Separator
from discord import ApplicationContext, OptionChoice
import google.generativeai as genai
import re
from typing_extensions import TypedDict
import json
load_dotenv()

dirname = os.path.dirname(__file__)
if not os.path.exists('caches'):
    os.mkdir('caches')

env = os.environ
separator = Separator(output_single_stem='Vocals',
                      output_dir=dirname+'/caches')
separator.load_model('Kim_Vocal_2.onnx')
bot = discord.Bot()
model = WhisperModel("turbo", device="auto", compute_type="int8")

genai.configure(api_key=env.get('GENAI_KEY', ''))
AIModel = genai.GenerativeModel("models/gemini-1.5-pro")

os.chdir('caches')

language_list = [
    OptionChoice("English", "en"),
    OptionChoice("Japanese", "ja"),
    OptionChoice("Korean", "ko"),
    OptionChoice("ZH-TW", "zh-tw"),
    OptionChoice("ZH-CN", "zh-cn"),
    OptionChoice("French", "fr"),
    OptionChoice("Spanish", "es"),
    OptionChoice("German", "de"),
    OptionChoice("Italian", "it"),
    OptionChoice("Russian", "ru"),
    OptionChoice("Portuguese", "pt"),
    OptionChoice("Dutch", "nl"),
    OptionChoice("Turkish", "tr"),
    OptionChoice("Arabic", "ar"),
    OptionChoice("Vietnamese", "vi"),
    OptionChoice("Thai", "th"),
    OptionChoice("Indonesian", "id"),
    OptionChoice("Hindi", "hi"),
    OptionChoice("Bengali", "bn"),
    OptionChoice("Filipino", "fil"),
    OptionChoice("Malay", "ms"),
    OptionChoice("Farsi", "fa"),
    OptionChoice("Urdu", "ur"),
    OptionChoice("Hebrew", "he"),
]

language_list_with_auto = language_list.copy()
language_list_with_auto.append(OptionChoice("Auto", "auto"))

language_list_with_blank = language_list.copy()
language_list_with_blank.append(OptionChoice("Blank", "blank"))

language_id_dict = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-tw": "ZH-TW",
    "zh-cn": "ZH-CN",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ru": "Russian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "hi": "Hindi",
}


class Language_Translate(TypedDict):
    original: str
    translated: str


@bot.slash_command(name="lyrics_to_lyrics", description="Translate lyrics to another language", options=[
    discord.Option(
        str, name='url', description="The lyrics you want to translate", required=True),
    discord.Option(
        str, name='to_language', description="The language you want to translate to", required=True, choices=language_list_with_blank, default="blank"),
    discord.Option(
        str, name='from_language', description="The language that audio is", required=False, choices=language_list_with_auto, default="Auto")
])
async def ltol(ctx: ApplicationContext, url: str, to_language: str = 'en', from_language: str = 'auto'):
    await ctx.defer()
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                    "preferredquality": "192",
                }
            ],
            "quiet": "True",
            'noplaylist': 'True',
            'outtmpl': f'{dirname}/caches/audio',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            info_dict = ydl.extract_info(url, download=False)
            video_id = info_dict.get('id', None)
            video_title = info_dict.get('title', None)

        text = []

        if not os.path.exists(video_id):
            os.mkdir(video_id)
        os.chdir(video_id)

        if (not os.path.exists(f"from_{from_language}.txt")):
            separator.separate(
                audio_file_path="../audio.m4a", primary_output_name="vocal")[0]

            if from_language == 'auto':
                result = model.transcribe(
                    '../vocal.wav')
            else:
                result = model.transcribe(
                    '../vocal.wav', language=from_language)
            result_language = result[1].language
            with open(f"from_{result_language}.txt", "w") as f:
                for segment in result[0]:
                    start = segment.start
                    end = segment.end
                    temp = f'{start//60:.0f}:{(start%60):.0f}-{end//60:.0f}:{end%60:.0f} : {segment.text} \n'
                    f.write(temp)
                    text.append(temp)
            if from_language == 'auto':
                with open(f"from_auto.txt", "w") as f:
                    for segment in result[0]:
                        start = segment.start
                        end = segment.end
                        temp = f'{start//60:.0f}:{(start%60):.0f}-{end//60:.0f}:{end%60:.0f} : {segment.text} \n'
                        f.write(temp)
                        text.append(temp)
        else:
            with open(f"from_{from_language}.txt", "r") as f:
                text = f.readlines()
        os.remove("../audio.m4a")
        source_language = result[1].language if from_language == 'auto' else from_language
        full_text = ''.join(text)
        length = len(full_text)
        if length > 2000 - 100:
            await ctx.respond(f"{video_title} - language {source_language} -  lyrics: ")
            for i in range(0, length, 2000-100):
                await ctx.respond(f"```{full_text[i:i+2000-10]}```")
        else:
            await ctx.respond(f"{video_title} - language {source_language} -  lyrics: \n```{full_text}```")

        if (to_language == 'blank'):
            os.chdir(dirname+'/caches')
            return
        if (not os.path.exists(f"to_{to_language}.txt")):
            response = AIModel.generate_content(
                "Translate "+re.sub(r"\d*:\d*-\d*:\d* : ", ' ', full_text)+" From " +
                language_id_dict[source_language]+" To " +
                language_id_dict[to_language] +
                " and return the translated text one by one",
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                    response_schema=list[Language_Translate]
                ))
            result = response.to_dict()
            result = result["candidates"][0]
            result = result["content"]["parts"][0]
            result = result["text"]
            with open(f"to_{to_language}.txt", "w") as f:
                f.write(result)
            result = json.loads(result)
        else:
            with open(f"to_{to_language}.txt", "r") as f:
                result = json.loads(f.read())
        translated = []
        for data in result:
            original = data["original"]
            translated_text = data["translated"]
            temp = f'\n {original} \n {translated_text} \n'
            translated.append(temp)
        length = len(''.join(translated))
        if length > 2000-100:
            await ctx.respond(f"Translated lyrics: ")
            for i in range(0, len(translated), 10):
                await ctx.send(''.join(translated[i:i+10]))
        else:
            await ctx.respond(f"Translated lyrics: \n{''.join(translated)}")
        os.chdir(dirname+'/caches')
        return
    except Exception as e:
        print(e.with_traceback(str))
        await ctx.respond(f"Error {e}", ephemeral=True)


bot.sync_commands()


@bot.listen(once=True)
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")
    print("Bot is ready")

# on Ctrl+C


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    bot.close()
    sys.exit(0)


bot.run(env.get('TOKEN', ''))
