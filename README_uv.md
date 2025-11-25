First create the conda env with
```
mm env create -n polymetis -f polymetis/environment_client.yml
```
Activate it.
The env contains C++ depedencies, but no python depedencies (notably not Pytorch).
You can then build everything by running `uv sync` within the conda environment.
You can rebuild and reinstall with `uv pip install --force-reinstall .`.

Alternatively, activate the conda env, source the uv env with Pytorch installed, and run
```
cmake -S polymetis -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
```
to build directly with CMake (may not be fully indentical to what uv builds though!).

One issue with the current setup is that there are two packages (`grpc` and `protobuf`) that are installed both with conda and uv and whos versions must match.
One could work around this by letting uv see system packages, though that would make wheel packaging a lot harder.
