@echo off
setlocal
pushd "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creation de l'environnement Python...
    where py >nul 2>&1
    if errorlevel 1 (
        python -m venv ".venv"
    ) else (
        py -3 -m venv ".venv"
    )
    if errorlevel 1 goto error
)
if not exist ".venv\caplets-dependances.ok" (
    echo Installation des bibliotheques...
    "%PY%" -m pip install --disable-pip-version-check -r "requirements.txt"
    if errorlevel 1 goto error
    type nul > ".venv\caplets-dependances.ok"
)
if "%~1"=="" (
    "%PY%" -u "reproduire.py" --choisir --ouvrir
) else (
    "%PY%" -u "reproduire.py" %* --ouvrir
)
set "STATUS=%errorlevel%"
if not "%STATUS%"=="0" pause
popd
exit /b %STATUS%

:error
echo Verifier le message ci-dessus. Utiliser Python 3.12 ou une version plus recente.
pause
popd
exit /b 1
