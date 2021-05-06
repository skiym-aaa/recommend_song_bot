from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
import os

app = Flask(__name__)

import pandas as pd
import recommend
# データの読み込み
df = pd.read_csv('AkimotoYasushi_all.csv', index_col=0)

#環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = os.environ["YOUR_CHANNEL_ACCESS_TOKEN"]
YOUR_CHANNEL_SECRET = os.environ["YOUR_CHANNEL_SECRET"]

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    query = event.message.text
    result_indexes = recommend.recommend(query)
    result_index = df.values[result_indexes[0]]
    song = result_index[0]
    artist = result_index[1]

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="「" + song + "」" + "\n" + artist))


if __name__ == "__main__":
#    app.run()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)