{
  description = "WCAG-compliant terminal color theme tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        cudaPackages = pkgs.cudaPackages;

        hexaColorSolver = pkgs.stdenv.mkDerivation {
          pname = "hexa-color-solver";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [
            cudaPackages.cuda_nvcc
            pkgs.cmake
            pkgs.ninja
          ];

          buildInputs = [
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cccl
            cudaPackages.libcurand
            pkgs.ftxui
          ];
        };

      in
      {
        packages = {
          default = hexaColorSolver;
          solver = hexaColorSolver;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (ps: [ ps.rich ps.pyte ]))
            pkgs.cmake
            pkgs.ninja
            pkgs.ftxui
            cudaPackages.cuda_nvcc
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cccl
            cudaPackages.libcurand
          ];

          shellHook = ''
            # Add nvidia driver libraries to path (NixOS)
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:''${LD_LIBRARY_PATH:-}

            echo "CUDA color optimizer dev shell"
            echo "Build with CMake:"
            echo "  mkdir -p build && cd build"
            echo "  cmake .. && cmake --build ."
          '';
        };

        devShells.rocm = pkgs.mkShell {
          buildInputs = [
            (pkgs.python3.withPackages (ps: [ ps.rich ps.pyte ]))
            pkgs.cmake
            pkgs.ninja
            pkgs.ftxui
            pkgs.rocmPackages.hipcc
            pkgs.rocmPackages.rocrand
            pkgs.rocmPackages.hiprand
            pkgs.rocmPackages.hipify
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.rocm-device-libs
            # cudaPackages.cuda_cudart
            # cudaPackages.cuda_cccl
            # cudaPackages.libcurand
          ];

          shellHook = ''
            export HIP_PLATFORM=amd
            export HIP_DEVICE_LIB_PATH=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode
            echo "ROCm color optimizer dev shell"
            echo "Build with CMake:"
            echo "  mkdir -p build && cd build"
            echo "  cmake .. -DUSE_ROCM=ON -DCMAKE_CXX_COMPILER=hipcc"
            echo "  cmake --build ."
          '';
        };
      }
    );
}
