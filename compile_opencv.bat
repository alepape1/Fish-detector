rem 1. (Opcional) Nivel de paralelismo
set CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%

rem 2. Configuración CMake (ajusta rutas a Python312)
"C:\Program Files\CMake\bin\cmake.exe" -H"C:\opencv_CUDA\opencv" ^
  -DOPENCV_EXTRA_MODULES_PATH="C:\opencv_CUDA\opencv_contrib\modules" ^
  -B"C:\opencv_CUDA\build" ^
  -G "Ninja Multi-Config" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DINSTALL_TESTS=ON ^
  -DINSTALL_C_EXAMPLES=ON ^
  -DBUILD_EXAMPLES=ON ^
  -DBUILD_opencv_world=ON ^
  -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON ^
  -DWITH_CUDA=ON ^
  -DCUDA_GENERATION=Auto ^
  -DBUILD_opencv_python3=ON ^
  -DPYTHON3_INCLUDE_DIR="C:/Users/Perfilador ResCoast/AppData/Local/Programs/Python/Python312/include" ^
  -DPYTHON3_LIBRARY="C:/Users/Perfilador ResCoast/AppData/Local/Programs/Python/Python312/libs/python312.lib" ^
  -DPYTHON3_EXECUTABLE="C:/Users/Perfilador ResCoast/AppData/Local/Programs/Python/Python312/python.exe" ^
  -DPYTHON3_NUMPY_INCLUDE_DIRS="C:/Users/Perfilador ResCoast/AppData/Local/Programs/Python/Python312/Lib/site-packages/numpy/_core/include" ^
  -DPYTHON3_PACKAGES_PATH="C:/Users/Perfilador ResCoast/AppData/Local/Programs/Python/Python312/Lib/site-packages/"

rem 3. Build e install
"C:\Program Files\CMake\bin\cmake.exe" --build "C:\opencv_CUDA\build" --target install --config Release
