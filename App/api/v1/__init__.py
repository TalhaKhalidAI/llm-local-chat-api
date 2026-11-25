from fastapi import APIRouter
 
from .Users import router as users_router
from .LlmChatEp import llmRouter
from .MFA import mfa_router
from .EmailOTP import email_otp_router

v1_router = APIRouter()
v1_router.include_router(users_router)
v1_router.include_router(llmRouter)
v1_router.include_router(mfa_router)
v1_router.include_router(email_otp_router)
