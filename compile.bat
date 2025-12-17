@echo off
set TEX_FILE=bare_jrnl

echo Cleaning auxiliary files...
del *.aux *.bbl *.blg *.log *.out *.spl *.pdf 2>nul

echo Step 1: Running pdflatex...
pdflatex -interaction=nonstopmode %TEX_FILE%.tex
if %errorlevel% neq 0 goto :error

echo Step 2: Running bibtex...
bibtex %TEX_FILE%
if %errorlevel% neq 0 goto :error

echo Step 3: Running pdflatex (resolving references)...
pdflatex -interaction=nonstopmode %TEX_FILE%.tex
if %errorlevel% neq 0 goto :error

echo Step 4: Running pdflatex (final pass)...
pdflatex -interaction=nonstopmode %TEX_FILE%.tex
if %errorlevel% neq 0 goto :error

echo Compilation complete. Output: %TEX_FILE%.pdf
goto :end

:error
echo.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo !!! COMPILATION FAILED !!!!!!!!!
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo Check %TEX_FILE%.log for details.
exit /b 1

:end
pause