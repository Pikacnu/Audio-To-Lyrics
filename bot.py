import discord
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import yt_dlp
from audio_separator.separator import Separator
from discord import ApplicationContext
load_dotenv()


dirname = os.path.dirname(__file__)
if not os.path.exists('caches'):
    os.mkdir('caches')

env = os.environ
separator = Separator(output_single_stem='Vocals',
                      output_dir=dirname+'/caches')
separator.load_model('Kim_Vocal_2.onnx')
bot = discord.Bot()
model = WhisperModel("base", device="auto", compute_type="int8")

os.chdir('caches')


@bot.slash_command(name="lyrics_to_lyrics", description="Translate lyrics to another language", options=[
    discord.Option(
        str, name='url', description="The lyrics you want to translate", required=True),
    discord.Option(
        str, name='language', description="The language you want to translate to", required=False)
])
async def ltol(ctx: ApplicationContext, url: str, language: str = None):
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
        if (not os.path.exists("audio.wav")):
            separator.separate(
                audio_file_path="../audio.m4a", primary_output_name="vocal")[0]

        if (not os.path.exists(f"from_{language}.txt")):
            result = model.transcribe('../vocal.wav', language=language) if language else model.transcribe(
                '../vocal.wav')
            result_language = result[1].language
            with open(f"from_{result_language}.txt", "w") as f:
                for segment in result[0]:
                    start = segment.start
                    end = segment.end
                    temp = f'{start//60:.0f}:{(start%60):.0f}-{end//60:.0f}:{end%60:.0f} : {segment.text} \n'
                    f.write(temp)
                    text.append(temp)
        else:
            with open(f"from_{language}.txt", "r") as f:
                text = f.readlines()

        length = len(''.join(text))
        os.chdir(dirname+'/caches')
        if length > 2000 - 100:
            await ctx.respond(f"{video_title} - language {result[1].language} -  lyrics: ")
            for i in range(0, length, 2000-100):
                await ctx.respond(f"```{''.join(text)[i:i+2000]}```")
        await ctx.respond(f"{video_title} - language {result[1].language} -  lyrics: \n```{''.join(text)}```")
        return
    except Exception as e:
        print(e)
        await ctx.respond(f"Error {e}", ephemeral=True)


bot.run(env.get('TOKEN', ''))
