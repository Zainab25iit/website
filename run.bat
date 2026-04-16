@echo off
echo Starting DeepVoiceAI Web Application...

:: Start Backend
start cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"

:: Start Frontend
start cmd /k "cd frontend && npm run dev"

echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
