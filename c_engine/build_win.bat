@echo off
REM Build the C backgammon engine as a DLL (requires GCC).
echo Building bg_engine.dll ...
gcc -O2 -shared -o bg_engine.dll bg_engine.c -Wall
if %errorlevel%==0 (
    echo Success: bg_engine.dll created.
) else (
    echo Build failed. Make sure GCC is installed and on PATH.
)
