import sys
import subprocess
import argparse


def check_python_version():
    """Check Python version and install appropriate dependencies."""
    version = sys.version_info
    if version.major == 3 and version.minor < 13:
        try:
            import pip
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "fcsparser>=0.1.2"
            ])
            print("Installed fcsparser for Python < 3.13")
        except Exception as e:
            print(f"Failed to install fcsparser: {e}")
            print("You may need to install it manually with: pip install fcsparser>=0.1.2")
    else:
        print("Python 3.13+ detected. fcsparser is optional.")
        print("If you need FCS file support, install with: pip install palantir[fcs]")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Palantir - Modeling continuous cell state and cell fate choices in single cell data"
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version information"
    )
    parser.add_argument(
        "--check-deps", action="store_true", help="Check dependencies and install as needed"
    )

    args = parser.parse_args()

    if args.version:
        from palantir.version import __version__
        print(f"Palantir version: {__version__}")
        return

    if args.check_deps:
        check_python_version()
        return

    if len(sys.argv) == 1:
        parser.print_help()


if __name__ == "__main__":
    main()