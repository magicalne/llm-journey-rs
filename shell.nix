{ pkgs ? import <nixpkgs> {} }:

let
  cudaPackages = pkgs.cudaPackages.overrideScope (final: prev: {
    cudnn = prev.cudnn_8_8;
  });
in
pkgs.mkShell {
  buildInputs = [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    # Add any other packages you need in your development environment
    pkgs.openssl
    pkgs.rustup
    pkgs.cmake
    pkgs.gcc
    pkgs.linuxPackages.nvidia_x11
  ];

  shellHook = ''
    export CUDNN_LIB=${cudaPackages.cudnn.dev}
    export CUDA_PATH=${cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH=${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:$LIBRARY_PATH
    export C_INCLUDE_PATH=${cudaPackages.cudatoolkit}/include:${cudaPackages.cudnn}/include:$C_INCLUDE_PATH
    export CPLUS_INCLUDE_PATH=${cudaPackages.cudatoolkit}/include:${cudaPackages.cudnn}/include:$CPLUS_INCLUDE_PATH
    
    echo "CUDA version: ${cudaPackages.cudatoolkit.version}"
    echo "cuDNN version: ${cudaPackages.cudnn.version}"
    echo "CUDA_PATH: $CUDA_PATH"
    echo "CUDNN_PATH: $CUDNN_PATH"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    
    # Check for cudnn.h
    if [ -f "$CUDNN_LIB/include/cudnn.h" ]; then
      echo "cudnn.h found in CUDNN_PATH"
    else
      echo "cudnn.h not found in CUDNN_PATH"
    fi
  '';
}
