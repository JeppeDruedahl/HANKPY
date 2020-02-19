set root=C:\Users\gmf123.UNICPH\AppData\Local\Continuum\anaconda3
call %root%\Scripts\activate.bat %root%
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "01 - Validation.ipynb"
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "02 - Timing.ipynb"
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "03 - Grids.ipynb"
call jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "04 - Calibration.ipynb"
pause
