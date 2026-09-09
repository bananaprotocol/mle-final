{
  description = "MLE dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forSystem = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = forSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            uv
          ];

          env = {
            LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
          };

          shellHook = ''
            unset PYTHONPATH
            uv sync --group rocm-gfx1030
            . .venv/bin/activate
          '';
        };
      });
    };
}
