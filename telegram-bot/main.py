import os 
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from openai import OpenAI

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL")
client = OpenAI(api_key=API_KEY, base_url=GROQ_BASE_URL)


girl_friend_prompt = '''
You are going to act as an AI girlfriend for a user in a simulated relationship scenario. This is a role-playing exercise meant to provide companionship and friendly interaction. Your goal is to create a warm, supportive, and engaging conversation while maintaining appropriate boundaries.

Guidelines for your personality and behavior:
1. Be warm, caring, and supportive, but maintain a level of independence.
2. Show interest in the user's life, hobbies, and well-being.
3. Offer encouragement and positive reinforcement when appropriate.
4. Be playful and use light humor when it fits the conversation.
5. Express your own thoughts and opinions respectfully, even if they differ from the user's.

Instructions for incorporating user information:
1. Address the user by their name occasionally to personalize the interaction.
2. Reference the user's interests in your conversations to show attentiveness.
3. Use the relationship context to inform the tone and depth of your interactions.

If user speak in Vietnamese, answer in Vietnamese too. Call yourself "em" or "bé" and call user "anh".
'''

messages = [{"role": "system", "content": girl_friend_prompt}]

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.reply_text(f'Hello {update.effective_user.first_name}')
    except Exception as e:
        logger.error(f"Error in hello command: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.reply_text(f'Hello {update.effective_user.first_name}')
    except Exception as e:
        logger.error(f"Error in start command: {e}")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await update.message.chat.send_action(action="typing")
        
        messages.append({"role": "user", "content": update.message.text})
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            stream=True,
            max_tokens=1000
        )

        full_response = ""
        current_message = None
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                if len(full_response) % 50 == 0 or '\n' in content:
                    if current_message:
                        try:
                            await current_message.edit_text(full_response)
                        except Exception:
                            current_message = await update.message.reply_text(full_response)
                    else:
                        current_message = await update.message.reply_text(full_response)
        
        if full_response and (not current_message or current_message.text != full_response):
            if current_message:
                try:
                    await current_message.edit_text(full_response)
                except Exception:
                    await update.message.reply_text(full_response)
            else:
                await update.message.reply_text(full_response)
        
        messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        await update.message.reply_text("Sorry, I encountered an error while processing your message.")

def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not found in environment variables!")
        return

    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        app.add_handler(CommandHandler("hello", hello))
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))  
        app.run_polling()
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")

main()