import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
    VOICE_ID = os.getenv('VOICE_ID')
    APP_PUBLIC_URL = os.getenv('APP_PUBLIC_URL')
    APP_SOCKET_URL = os.getenv('APP_SOCKET_URL')
    APP_PUBLIC_GATHER_URL = f"{APP_PUBLIC_URL}/gather"
    APP_PUBLIC_EVENT_URL = f"{APP_PUBLIC_URL}/event"
    COMPANY_NAME = os.getenv('COMPANY_NAME')
    COMPANY_BUSINESS = os.getenv('COMPANY_BUSINESS')
    COMPANY_PRODUCT_SERVICES = os.getenv('COMPANY_PRODUCTS_SERVICES')
    CONVERSATION_PURPOSE = os.getenv('CONVERSATION_PURPOSE')
    AISALESAGENT_NAME = os.getenv('AISALESAGENT_NAME')
