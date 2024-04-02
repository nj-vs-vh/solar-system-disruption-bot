from telebot import TeleBot
from pathlib import Path

from solar_system.config import load_config

config = load_config()
bot = TeleBot(token=config['telegram_api_token'])


def send_to_channel(text: str, video: Path):
    chat_id = config['telegram_channel_id']
    video_stream = open(video, 'rb')
    print("Sending video file")
    bot.send_video(chat_id=chat_id, video=video_stream, caption=text)
    print("DONE!")
